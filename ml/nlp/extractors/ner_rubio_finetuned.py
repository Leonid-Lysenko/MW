# ml/nlp/extractors/ner_rubio_finetuned.py
import os
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .base import SymptomExtractor
from django.db import connection

# HuggingFace ID модели
HF_MODEL_ID = "velto006/rubio-ner-finetuned-74"


class RuBioRobertaNERExtractor(SymptomExtractor):
    """
    Экстрактор симптомов на основе дообученной RuBioRoBERTa.
    Обучен на 74 симптомах (только 3 класса: O, B-SYMP, I-SYMP).
    Модель загружается с HuggingFace Hub при отсутствии локальной копии.
    """
    
    def __init__(self, model_path: Optional[str] = None, synonyms_path: Optional[str] = None):
        """Инициализирует экстрактор."""
        super().__init__()
        
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if synonyms_path is None:
            synonyms_path = os.path.join(current_dir, 'data', 'synonyms', 'synonyms_74.json')
        
        # Определяем путь к модели
        if model_path is None:
            local_path = os.path.join(current_dir, 'models', 'rubio_ner_finetuned_74')
            
            # Проверяем наличие локальной модели
            if os.path.exists(os.path.join(local_path, 'model.safetensors')):
                model_path = local_path
            else:
                # Загружаем с HuggingFace
                from huggingface_hub import snapshot_download
                model_path = snapshot_download(
                    repo_id=HF_MODEL_ID,
                    local_dir=local_path,
                    local_dir_use_symlinks=False,
                    ignore_patterns=["*.pt", "optimizer-*.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"]
                )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        self.symptom_map = self._load_symptom_map()
        self.synonyms_map = self._load_synonyms(synonyms_path)
    
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
        """Объединяет BPE-токены в цельные слова."""
        merged = []
        current_word = ""
        current_label = None
        current_scores = []
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if token in ['<s>', '</s>', '<pad>']:
                continue
            
            score = probs[i] if i < len(probs) else 0.5
            
            if token.startswith('##'):
                current_word += token[2:]
                current_scores.append(score)
                if current_label is None:
                    current_label = label
            else:
                if current_word and current_label:
                    avg_score = sum(current_scores) / len(current_scores)
                    merged.append({
                        'text': current_word,
                        'label': current_label,
                        'score': avg_score
                    })
                
                clean_token = token.replace('Ġ', '')
                try:
                    decoded = self.tokenizer.convert_tokens_to_string([clean_token])
                    current_word = decoded
                except:
                    current_word = clean_token
                
                current_label = label
                current_scores = [score]
        
        if current_word and current_label:
            avg_score = sum(current_scores) / len(current_scores)
            merged.append({
                'text': current_word,
                'label': current_label,
                'score': avg_score
            })
        
        return merged
    
    def _group_entities(self, words: List[Dict]) -> List[Dict]:
        """Группирует слова в сущности по BIO-разметке."""
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
        
        if text_lower in self.symptom_map:
            return text, self.symptom_map[text_lower]
        
        if text_lower in self.synonyms_map:
            canon_name, symptom_id = self.synonyms_map[text_lower]
            return canon_name, symptom_id
        
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
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        preds = predictions[0].tolist()
        labels = [self.id2label[p] for p in preds]
        token_probs = [probs[0][i][p].item() for i, p in enumerate(preds)]
        
        words = self._merge_bpe_tokens(tokens, labels, token_probs)
        entities = self._group_entities(words)
        
        symptoms = []
        for entity in entities:
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