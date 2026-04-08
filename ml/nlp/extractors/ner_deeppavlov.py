# ml/nlp/extractors/ner_deeppavlov.py
from typing import List, Dict, Any, Tuple
from .base import SymptomExtractor
from deeppavlov import build_model, configs

class DeepPavlovNERExtractor(SymptomExtractor):
    """
    Экстрактор симптомов на основе DeepPavlov RuBERT NER.
    
    Выступает в роли baseline для NER-подходов.
    Модель не специализирована на медицине и не дообучалась.
    """
    
    def __init__(self):
        """Загружает предобученную NER-модель."""
        super().__init__()
        print("Загрузка модели DeepPavlov RuBERT NER...")
        # Загружаем модель NER
        self.model = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
        print("✓ Модель загружена")
        
        # Маппинг меток NER на русские названия
        self.entity_types = {
            'PER': 'человек',
            'LOC': 'локация',
            'ORG': 'организация',
            'GPE': 'геополитическая единица',
            'DATE': 'дата',
            'TIME': 'время',
            'MONEY': 'деньги',
            'PERCENT': 'процент',
            'FAC': 'сооружение',
            'PRODUCT': 'продукт',
            'EVENT': 'событие',
            'LAW': 'закон',
            'LANGUAGE': 'язык',
            'WORK_OF_ART': 'произведение искусства',
        }
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает сущности из текста с помощью NER-модели.
        
        Args:
            text: входной текст
            
        Returns:
            Всегда возвращает пустой список, так как модель не находит симптомы.
            Это методологически важно для baseline.
        """
        if not text:
            return []
        
        # Модель возвращает токены и метки
        tokens, tags = self.model([text])
        
        # Группируем токены в сущности (только для отладки)
        entities = self._extract_entities(tokens[0], tags[0])
        
        # В консоль выводим, что нашла модель (для понимания)
        if entities:
            print(f"\n[DeepPavlov] В тексте '{text[:30]}...' найдены сущности:")
            for ent in entities:
                print(f"  - {ent['text']} ({ent['type_name']})")
        
        # Возвращаем пустой список — симптомы не распознаются
        return []
    
    def _extract_entities(self, tokens: List[str], tags: List[str]) -> List[Dict]:
        """
        Группирует токены в сущности по BIO-разметке.
        """
        entities = []
        current_entity = None
        
        for token, tag in zip(tokens, tags):
            if tag.startswith('B-'):
                # Начало новой сущности
                if current_entity:
                    entities.append(current_entity)
                entity_type = tag[2:]
                current_entity = {
                    'text': token,
                    'type': entity_type,
                    'type_name': self.entity_types.get(entity_type, entity_type)
                }
            elif tag.startswith('I-') and current_entity:
                # Продолжение текущей сущности
                current_entity['text'] += ' ' + token
            else:
                # Конец сущности
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities