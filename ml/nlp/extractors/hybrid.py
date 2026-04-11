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
# from .semantic_search import SemanticSearchExtractor
# from .ner_rubio_finetuned import RuBioRobertaNERExtractor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def _load_rule_based(synonyms_path: str):
    """Загрузка rule-based модуля."""
    return RuleBasedExtractor(synonyms_path=synonyms_path)


class HybridExtractor(SymptomExtractor):
    """
    Гибридный экстрактор симптомов.
    """
    
    def __init__(self, 
                 ner_model_path: Optional[str] = None,
                 semantic_threshold_common: float = 0.85,
                 semantic_threshold_rare: float = 0.75,
                 ner_confidence_threshold: float = 0.75,
                 rule_based_synonyms_path: Optional[str] = None):
        super().__init__()
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if rule_based_synonyms_path is None:
            rule_based_synonyms_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_updated.json')
        
        print("[Hybrid] Инициализация компонентов...")
        
        self.rule = RuleBasedExtractor(synonyms_path=rule_based_synonyms_path)
        print(" Rule-based загружен")
        
        self.semantic_threshold_common = semantic_threshold_common
        self.semantic_threshold_rare = semantic_threshold_rare
        self.ner_confidence_threshold = ner_confidence_threshold
        
        self.top74_symptom_ids = self._load_top74_symptom_ids()
        
        print(f"[Hybrid] Готов. Топ-74 симптомов: {len(self.top74_symptom_ids)}")
    
    def _load_top74_symptom_ids(self) -> Set[int]:
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
        return symptom_id in self.top74_symptom_ids
    
    def _normalize_name(self, name: str) -> str:
        return ' '.join(name.lower().split())
    
    def _check_negation_with_rule(self, text: str, symptom_name: str) -> str:
        return self.rule.check_symptom_negation(text, symptom_name)
    
    def _is_symptom_in_text(self, text: str, symptom_name: str) -> bool:
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
        if not text:
            return []
        
        rule_results = self.rule.extract(text)
        
        added_symptoms: Set[str] = set()
        hybrid_results = []
        
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
        
        hybrid_results.sort(key=lambda x: x['confidence'], reverse=True)
        return hybrid_results
    
    def extract_with_details(self, text: str) -> Dict[str, Any]:
        return {
            'text': text,
            'rule_based': self.rule.extract(text),
            'ner': [],
            'semantic': [],
            'hybrid': self.extract(text)
        }