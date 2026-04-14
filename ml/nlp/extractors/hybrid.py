# ml/nlp/extractors/hybrid.py
import os
import sys
import io
import re
import json
from typing import List, Dict, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base import SymptomExtractor
from .rule_based import RuleBasedExtractor

# Импортируем дополнительные модули только если они нужны
try:
    from .semantic_search import SemanticSearchExtractor
    from .ner_rubio_finetuned import RuBioRobertaNERExtractor
    FULL_MODE_AVAILABLE = True
except ImportError:
    FULL_MODE_AVAILABLE = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Определяем режим работы по переменной окружения
DEBUG = os.environ.get("DEBUG", "True") == "True"


def _load_rule_based(synonyms_path: str):
    """Загрузка rule-based модуля."""
    return RuleBasedExtractor(synonyms_path=synonyms_path)


def _load_ner(model_path: str, synonyms_path: str):
    """Загрузка NER модуля."""
    if FULL_MODE_AVAILABLE:
        return RuBioRobertaNERExtractor(model_path, synonyms_path)
    return None


def _load_semantic(threshold: float):
    """Загрузка семантического модуля."""
    if FULL_MODE_AVAILABLE:
        return SemanticSearchExtractor(
            model_name='e5-large' if DEBUG else 'intfloat/multilingual-e5-small',
            confidence_threshold=threshold
        )
    return None


class HybridExtractor(SymptomExtractor):
    """
    Гибридный экстрактор симптомов.
    
    В режиме DEBUG=True загружает все компоненты (NER, семантика, rule-based).
    В режиме DEBUG=False (продакшен) загружает только rule-based.
    """
    
    def __init__(self, 
                 ner_model_path: Optional[str] = None,
                 semantic_threshold_common: float = 0.85,
                 semantic_threshold_rare: float = 0.75,
                 ner_confidence_threshold: float = 0.75,
                 rule_based_synonyms_path: Optional[str] = None):
        super().__init__()
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Путь к модели NER (только для локальной разработки)
        if ner_model_path is None:
            ner_model_path = os.path.join(current_dir, 'models', 'rubio_ner_finetuned_74', 'checkpoint-1431')
        
        # Путь к словарю синонимов
        if rule_based_synonyms_path is None:
            rule_based_synonyms_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_updated.json')
        
        print("[Hybrid] Инициализация компонентов...")
        
        # Rule-based всегда загружается
        self.rule = RuleBasedExtractor(synonyms_path=rule_based_synonyms_path)
        print(" Rule-based загружен")
        
        if DEBUG:
            # Локальная разработка: загружаем все компоненты
            print(" DEBUG mode: loading full components...")
            self.ner = _load_ner(ner_model_path, rule_based_synonyms_path)
            self.semantic = _load_semantic(semantic_threshold_rare)
            if self.ner:
                print(" NER загружен")
            if self.semantic:
                print(" Семантический поиск загружен")
        else:
            # Продакшен (Render): только rule-based
            print(" PRODUCTION mode: rule-based only")
            self.ner = None
            self.semantic = None
        
        self.semantic_threshold_common = semantic_threshold_common
        self.semantic_threshold_rare = semantic_threshold_rare
        self.ner_confidence_threshold = ner_confidence_threshold
        
        # Загружаем ID симптомов из топ-74 (нужно для NER)
        self.top74_symptom_ids = self._load_top74_symptom_ids()
        
        print(f"[Hybrid] Готов. Топ-74 симптомов: {len(self.top74_symptom_ids)}")
    
    def _load_top74_symptom_ids(self) -> Set[int]:
        """Загружает множество ID симптомов из топ-74."""
        top74_ids = set()
        try:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synonyms_74_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_74.json')
            
            with open(synonyms_74_path, 'r', encoding='utf-8') as f:
                synonyms_74 = json.load(f)
            
            symptom_map = self.rule.canonical_to_id if hasattr(self.rule, 'canonical_to_id') else {}
            
            for canonical_name in synonyms_74.keys():
                symptom_id = symptom_map.get(canonical_name)
                if symptom_id:
                    top74_ids.add(symptom_id)
                    
        except Exception as e:
            print(f" Ошибка загрузки топ-74 симптомов: {e}")
        
        return top74_ids
    
    def _is_symptom_in_top74(self, symptom_id: int) -> bool:
        """Проверяет, входит ли симптом в топ-74."""
        return symptom_id in self.top74_symptom_ids
    
    def _normalize_name(self, name: str) -> str:
        """Нормализует название симптома для сравнения."""
        return ' '.join(name.lower().split())
    
    def _check_negation_with_rule(self, text: str, symptom_name: str) -> str:
        """Проверяет отрицание через rule-based."""
        return self.rule.check_symptom_negation(text, symptom_name)
    
    def _is_symptom_in_text(self, text: str, symptom_name: str) -> bool:
        """Проверяет вхождение симптома в текст."""
        text_lower = text.lower()
        symptom_lower = symptom_name.lower()
        symptom_words = symptom_lower.split()
        
        if len(symptom_words) == 1:
            if re.search(r'\b' + re.escape(symptom_lower) + r'\b', text_lower):
                return True
        else:
            if symptom_lower in text_lower:
                return True
        return False
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает симптомы из текста."""
        if not text:
            return []
        
        # Rule-based результаты (основа)
        rule_results = self.rule.extract(text)
        
        added_symptoms: Set[str] = set()
        hybrid_results = []
        
        # Добавляем rule-based
        for symptom in rule_results:
            name = symptom['canonical_name']
            norm_name = self._normalize_name(name)
            hybrid_results.append({
                'symptom_id': symptom['symptom_id'],
                'canonical_name': name,
                'status': symptom['status'],
                'confidence': 1.0,
                'source': 'rule_based'
            })
            added_symptoms.add(norm_name)
        
        # Если в режиме DEBUG и есть дополнительные модули — добавляем их
        if DEBUG and self.ner and self.semantic:
            ner_results = self.ner.extract(text)
            semantic_results = self.semantic.extract(text, top_k=10)
            
            # NER (только для топ-74 симптомов)
            for symptom in ner_results:
                name = symptom['canonical_name']
                norm_name = self._normalize_name(name)
                
                if norm_name in added_symptoms:
                    continue
                
                if not self._is_symptom_in_top74(symptom['symptom_id']):
                    continue
                
                if symptom['confidence'] < self.ner_confidence_threshold:
                    continue
                
                if not self._is_symptom_in_text(text, name):
                    continue
                
                status = self._check_negation_with_rule(text, name)
                
                hybrid_results.append({
                    'symptom_id': symptom['symptom_id'],
                    'canonical_name': name,
                    'status': status,
                    'confidence': symptom['confidence'],
                    'source': 'ner'
                })
                added_symptoms.add(norm_name)
            
            # Semantic (разные пороги)
            for symptom in semantic_results:
                name = symptom['canonical_name']
                norm_name = self._normalize_name(name)
                
                if norm_name in added_symptoms:
                    continue
                
                symptom_id = symptom['symptom_id']
                is_common = self._is_symptom_in_top74(symptom_id)
                
                threshold = self.semantic_threshold_common if is_common else self.semantic_threshold_rare
                
                if symptom['confidence'] < threshold:
                    continue
                
                if not self._is_symptom_in_text(text, name):
                    continue
                
                status = self._check_negation_with_rule(text, name)
                
                hybrid_results.append({
                    'symptom_id': symptom_id,
                    'canonical_name': name,
                    'status': status,
                    'confidence': symptom['confidence'],
                    'source': 'semantic'
                })
                added_symptoms.add(norm_name)
        
        hybrid_results.sort(key=lambda x: x['confidence'], reverse=True)
        return hybrid_results
    
    def extract_with_details(self, text: str) -> Dict[str, Any]:
        """Возвращает результаты каждого модуля для отладки."""
        return {
            'text': text,
            'rule_based': self.rule.extract(text),
            'ner': self.ner.extract(text) if self.ner else [],
            'semantic': self.semantic.extract(text, top_k=10) if self.semantic else [],
            'hybrid': self.extract(text)
        }