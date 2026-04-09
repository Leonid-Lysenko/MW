# ml/nlp/extractors/ner_rubio_finetuned.py
import os
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .base import SymptomExtractor
from django.db import connection

class RuBioRobertaNERExtractor(SymptomExtractor):
    """
    Экстрактор симптомов на основе дообученной RuBioRoBERTa.
    Обучен на 74 симптомах (только 3 класса: O, B-SYMP, I-SYMP).
    """
    
    def __init__(self, model_path: Optional[str] = None, synonyms_path: Optional[str] = None):
        """Инициализирует экстрактор."""
        super().__init__()
        
        if model_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(current_dir, 'models', 'rubio_ner_finetuned_74', 'checkpoint-1431')
        
        if synonyms_path is None:
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synonyms_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_74.json')
        
        # print(f"Загрузка дообученной модели из {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        # Загружаем маппинг симптомов из БД (все 304)
        self.symptom_map = self._load_symptom_map()
        
        # Загружаем словарь синонимов (только для 74 симптомов)
        self.synonyms_map = self._load_synonyms(synonyms_path)
        
        # print(f" Модель загружена. Метки: {list(self.id2label.values())}")
        # print(f" Загружено {len(self.symptom_map)} симптомов из БД")
        # print(f" Загружено {len(self.synonyms_map)} синонимов (74 симптома)")
    
    def _load_symptom_map(self) -> Dict[str, int]:
        """Загружает маппинг названий симптомов в их ID из БД."""
        symptom_map = {}
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, name FROM diagnosis_symptom")
                rows = cursor.fetchall()
                for symptom_id, name in rows:
                    symptom_map[name.lower()] = symptom_id
                    symptom_map[name] = symptom_id
        except Exception as e:
            print(f" Ошибка загрузки симптомов из БД: {e}")
        return symptom_map
    
    def _load_synonyms(self, synonyms_path: str) -> Dict[str, tuple]:
        """Загружает словарь синонимов (только для 74 симптомов)."""
        synonyms_map = {}
        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            for canonical_name, synonyms in synonyms_data.items():
                symptom_id = self.symptom_map.get(canonical_name.lower())
                if symptom_id:
                    for synonym in synonyms:
                        synonyms_map[synonym.lower()] = (canonical_name, symptom_id)
        except Exception as e:
            print(f" Ошибка загрузки синонимов: {e}")
        return synonyms_map
    
    def _merge_bpe_tokens(self, tokens: List[str], labels: List[str], probs: List[float]) -> List[Dict]:
        """
        Объединяет BPE-токены в цельные слова.
        """
        merged = []
        current_word = ""
        current_label = None
        current_scores = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            score = probs[i] if i < len(probs) else 0.5
            
            # Токен с ## - это продолжение
            if token.startswith('##'):
                current_word += token[2:]
                current_scores.append(score)
                if current_label is None:
                    current_label = label
            else:
                # Сохраняем предыдущее слово
                if current_word and current_label:
                    avg_score = sum(current_scores) / len(current_scores)
                    merged.append({
                        'text': current_word,
                        'label': current_label,
                        'score': avg_score
                    })
                
                # Начинаем новое слово
                # Убираем символ Ġ (пробел)
                clean_token = token.replace('Ġ', '')
                
                # Пробуем декодировать через токенизатор
                try:
                    decoded = self.tokenizer.convert_tokens_to_string([clean_token])
                    current_word = decoded
                except:
                    current_word = clean_token
                
                current_label = label
                current_scores = [score]
        
        # Добавляем последнее слово
        if current_word and current_label:
            avg_score = sum(current_scores) / len(current_scores)
            merged.append({
                'text': current_word,
                'label': current_label,
                'score': avg_score
            })
        
        return merged
    
    def _group_entities(self, words: List[Dict]) -> List[Dict]:
        """
        Группирует слова в сущности по BIO-разметке.
        """
        entities = []
        current_entity = None
        
        for word in words:
            label = word['label']
            
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': word['text'],
                    'label': label,
                    'score': word['score'],
                    'words': [word['text']]
                }
            elif label.startswith('I-') and current_entity:
                current_entity['text'] += ' ' + word['text']
                current_entity['words'].append(word['text'])
                current_entity['score'] = (current_entity['score'] + word['score']) / 2
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _find_symptom_by_text(self, text: str) -> tuple:
        """Находит симптом по тексту, используя словарь синонимов."""
        text_lower = text.lower().strip()
        
        # 1. Точное совпадение с каноническим названием
        if text_lower in self.symptom_map:
            return text, self.symptom_map[text_lower]
        
        # 2. Поиск по синонимам
        if text_lower in self.synonyms_map:
            canon_name, symptom_id = self.synonyms_map[text_lower]
            return canon_name, symptom_id
        
        # 3. Поиск по вхождению
        for canon_name, sid in self.symptom_map.items():
            if isinstance(canon_name, str):
                canon_lower = canon_name.lower()
                if len(text_lower) > 2 and text_lower in canon_lower:
                    return canon_name, sid
        
        return text, None
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает симптомы из текста."""
        if not text:
            return []
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Получаем BPE-токены
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        preds = predictions[0].tolist()
        labels = [self.id2label[p] for p in preds]
        token_probs = [probs[0][i][p].item() for i, p in enumerate(preds)]
        
        # Объединяем BPE-токены в слова
        words = self._merge_bpe_tokens(tokens, labels, token_probs)
        
        # Группируем слова в сущности
        entities = self._group_entities(words)
        
        symptoms = []
        for entity in entities:
            # Только B-SYMP и I-SYMP
            if entity['label'] in ['B-SYMP', 'I-SYMP']:
                status = 'present'
            else:
                continue
            
            canon_name, symptom_id = self._find_symptom_by_text(entity['text'])
            
            if symptom_id is None:
                continue
            
            symptoms.append({
                'symptom_id': symptom_id,
                'canonical_name': canon_name,
                'status': status,
                'confidence': entity['score'],
                'matched_text': entity['text']
            })
        
        return symptoms