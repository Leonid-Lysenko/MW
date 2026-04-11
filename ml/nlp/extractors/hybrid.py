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
from .semantic_search import SemanticSearchExtractor
# from .ner_rubio_finetuned import RuBioRobertaNERExtractor  # <-- ЗАКОММЕНТИРОВАНО

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def _load_rule_based(synonyms_path: str):
    """Загрузка rule-based модуля."""
    return RuleBasedExtractor(synonyms_path=synonyms_path)


# def _load_ner(model_path: str, synonyms_path: str):  # <-- ЗАКОММЕНТИРОВАНО
#     """Загрузка NER модуля."""
#     return RuBioRobertaNERExtractor(model_path, synonyms_path)


def _load_semantic(threshold: float):
    """Загрузка семантического модуля."""
    return SemanticSearchExtractor(
        model_name='intfloat/multilingual-e5-small',  # <-- ИЗМЕНЕНО с e5-large на e5-small
        confidence_threshold=threshold
    )


class HybridExtractor(SymptomExtractor):
    """
    Гибридный экстрактор симптомов.
    
    Комбинирует три подхода с приоритетом:
    1. Rule-based - основа (высокая точность, понимает отрицания)
    2. NER - дополняет rule-based для частотных симптомов (топ-74)
    3. Семантический поиск - для редких симптомов (вне топ-74)
    
    Загрузка компонентов выполняется параллельно для ускорения инициализации.
    """
    
    def __init__(self, 
                 ner_model_path: Optional[str] = None,
                 semantic_threshold_common: float = 0.85,
                 semantic_threshold_rare: float = 0.75,
                 ner_confidence_threshold: float = 0.75,
                 rule_based_synonyms_path: Optional[str] = None):
        """Инициализирует гибридный экстрактор с параллельной загрузкой."""
        super().__init__()
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Путь к новой модели (обученной на 74 симптомах, 3 класса)
        if ner_model_path is None:
            ner_model_path = os.path.join(current_dir, 'models', 'rubio_ner_finetuned_74', 'checkpoint-1431')
        
        # Путь к обновлённому словарю синонимов
        if rule_based_synonyms_path is None:
            rule_based_synonyms_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_updated.json')
        
        print("[Hybrid] Инициализация компонентов (параллельная загрузка)...")
        
        # Параллельная загрузка трёх компонентов
        self.rule = None
        self.ner = None  # <-- ОСТАВЛЕН, НО БУДЕТ None
        self.semantic = None
        
        with ThreadPoolExecutor(max_workers=2) as executor:  # <-- ИЗМЕНЕНО с 3 на 2
            # Запускаем задачи параллельно
            future_rule = executor.submit(_load_rule_based, rule_based_synonyms_path)
            # future_ner = executor.submit(_load_ner, ner_model_path, rule_based_synonyms_path)  # <-- ЗАКОММЕНТИРОВАНО
            future_semantic = executor.submit(_load_semantic, semantic_threshold_rare)
            
            # Собираем результаты по мере завершения
            for future in as_completed([future_rule, future_semantic]):  # <-- УБРАН future_ner
                try:
                    result = future.result()
                    # Определяем, какой компонент загрузился
                    if isinstance(result, RuleBasedExtractor):
                        self.rule = result
                        print(" Rule-based загружен")
                    # elif isinstance(result, RuBioRobertaNERExtractor):  # <-- ЗАКОММЕНТИРОВАНО
                    #     self.ner = result
                    #     print(" NER загружен")
                    elif isinstance(result, SemanticSearchExtractor):
                        self.semantic = result
                        print(" Семантический поиск загружен")
                except Exception as e:
                    print(f" Ошибка загрузки компонента: {e}")
        
        # Проверяем, что все компоненты загружены
        # if self.rule is None or self.ner is None or self.semantic is None:  # <-- СТАРОЕ
        if self.rule is None or self.semantic is None:  # <-- ИЗМЕНЕНО (убрана проверка NER)
            raise RuntimeError("Не удалось загрузить все компоненты гибридного экстрактора")
        
        self.semantic_threshold_common = semantic_threshold_common
        self.semantic_threshold_rare = semantic_threshold_rare
        self.ner_confidence_threshold = ner_confidence_threshold
        
        # Загружаем ID симптомов из топ-74
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
            
            # Получаем маппинг симптомов из rule-based
            symptom_map = self.rule.canonical_to_id if hasattr(self.rule, 'canonical_to_id') else {}
            
            # Если rule-based не дал маппинг, загружаем из БД
            if not symptom_map:
                from django.db import connection
                with connection.cursor() as cursor:
                    cursor.execute("SELECT id, name FROM diagnosis_symptom")
                    for symptom_id, name in cursor.fetchall():
                        symptom_map[name] = symptom_id
            
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
        """
        Использует rule-based для проверки отрицания конкретного симптома.
        Rule-based ищет в тексте маркеры отрицания рядом с симптомом.
        """
        return self.rule.check_symptom_negation(text, symptom_name)
    
    def _is_symptom_in_text(self, text: str, symptom_name: str) -> bool:
        """
        Проверяет, встречается ли симптом в тексте (точное вхождение слова/фразы).
        Используется для фильтрации ложных срабатываний.
        """
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
        """
        Извлекает симптомы с приоритетом:
        1. Rule-based (основа, все симптомы)
        2. NER (только для частотных симптомов, порог 0.75)
        3. Семантический поиск (разные пороги для частотных и редких)
        """
        if not text:
            return []
        
        # Шаг 1: Получаем результаты
        rule_results = self.rule.extract(text)
        # ner_results = self.ner.extract(text) if self.ner else []  # <-- ЗАКОММЕНТИРОВАНО
        ner_results = []  # <-- ДОБАВЛЕНО (заглушка вместо NER)
        semantic_results = self.semantic.extract(text, top_k=10)
        
        added_symptoms: Set[str] = set()
        hybrid_results = []
        
        # Шаг 2: Rule-based (всегда)
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
        
        # Шаг 3: NER (только для частотных симптомов из топ-74) - ВРЕМЕННО ОТКЛЮЧЕН
        # for symptom in ner_results:  # <-- ЗАКОММЕНТИРОВАНО
        #     name = symptom['canonical_name']
        #     norm_name = self._normalize_name(name)
        #     
        #     if norm_name in added_symptoms:
        #         continue
        #     
        #     if not self._is_symptom_in_top74(symptom['symptom_id']):
        #         continue
        #     
        #     if symptom['confidence'] < self.ner_confidence_threshold:
        #         continue
        #     
        #     if not self._is_symptom_in_text(text, name):
        #         continue
        #     
        #     status = self._check_negation_with_rule(text, name)
        #     
        #     hybrid_results.append({
        #         'symptom_id': symptom['symptom_id'],
        #         'canonical_name': name,
        #         'status': status,
        #         'confidence': symptom['confidence'],
        #         'source': 'ner'
        #     })
        #     added_symptoms.add(norm_name)
        
        # Шаг 4: Semantic (разные пороги для частотных и редких)
        for symptom in semantic_results:
            name = symptom['canonical_name']
            norm_name = self._normalize_name(name)
            
            if norm_name in added_symptoms:
                continue
            
            symptom_id = symptom['symptom_id']
            is_common = self._is_symptom_in_top74(symptom_id)
            
            if is_common:
                threshold = self.semantic_threshold_common
            else:
                threshold = self.semantic_threshold_rare
            
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
            'ner': [],  # <-- ИЗМЕНЕНО (NER отключен)
            'semantic': self.semantic.extract(text, top_k=10),
            'hybrid': self.extract(text)
        }