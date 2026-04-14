# ml/nlp/extractors/semantic_search.py
import numpy as np
import os
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from .base import SymptomExtractor
import torch

# Определяем режим работы по переменной окружения
DEBUG = os.environ.get("DEBUG", "True") == "True"

# Импортируем connection только если нужна БД (только для локальной разработки)
if DEBUG:
    from django.db import connection


class SemanticSearchExtractor(SymptomExtractor):
    """
    Экстрактор симптомов на основе семантического поиска.
    Использует эмбеддинги и pgvector для поиска ближайших симптомов.
    
    ВНИМАНИЕ: Этот модуль используется ТОЛЬКО в режиме DEBUG=True (локально).
    На Render (DEBUG=False) он не загружается.
    """
    
    def __init__(self, model_name: str = 'e5-large', confidence_threshold: float = 0.6):
        """
        Args:
            model_name: 'e5-large' или 'rubioroberta'
            confidence_threshold: порог уверенности (0.0-1.0)
        """
        super().__init__()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
        
    def _load_model(self):
        """Загружает соответствующую модель."""
        if self.model_name == 'e5-large':
            return SentenceTransformer('intfloat/multilingual-e5-large')
        elif self.model_name == 'rubioroberta':
            # Для RuBioRoBERTa нужно эмбеддинги считать отдельно
            self.tokenizer = AutoTokenizer.from_pretrained('alexyalunin/RuBioRoBERTa')
            model = AutoModel.from_pretrained('alexyalunin/RuBioRoBERTa')
            model.eval()
            return model
        else:
            raise ValueError(f"Неизвестная модель: {self.model_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг для текста."""
        if self.model_name == 'e5-large':
            # E5 требует префикс "query:" для поиска
            text = f"query: {text}"
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
        elif self.model_name == 'rubioroberta':
            # Для RuBioRoBERTa используем mean pooling
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.squeeze().numpy()
                # Нормализуем
                embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
    
    def extract(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Извлекает симптомы из текста через семантический поиск.
        """
        if not text:
            return []
        
        # Получаем эмбеддинг текста
        query_embedding = self._get_embedding(text)
        embedding_str = '[' + ','.join(f"{x:.8f}" for x in query_embedding) + ']'
        
        # Ищем ближайшие симптомы в БД
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    se.symptom_id,
                    ds.name as canonical_name,
                    1 - (se.embedding <=> %s::vector) as similarity
                FROM symptom_embeddings se
                JOIN diagnosis_symptom ds ON se.symptom_id = ds.id
                WHERE se.model_name = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, [embedding_str, self.model_name, top_k])
            
            results = cursor.fetchall()
        
        # Фильтруем по порогу и форматируем результат
        symptoms = []
        for symptom_id, canonical_name, similarity in results:
            if similarity >= self.confidence_threshold:
                symptoms.append({
                    'symptom_id': symptom_id,
                    'canonical_name': canonical_name,
                    'status': 'present',  # семантический поиск не понимает отрицания
                    'confidence': float(similarity),
                    'matched_text': text
                })
        
        return symptoms
    
    def find_similar(self, symptom_name: str, exclude: Set[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Находит симптомы, похожие на указанный.
        Используется для рекомендаций в интерфейсе.
        """
        if exclude is None:
            exclude = set()
        
        if not symptom_name:
            return []
        
        try:
            # Получаем эмбеддинг симптома
            embedding = self._get_embedding(symptom_name)
            embedding_str = '[' + ','.join(f"{x:.8f}" for x in embedding) + ']'
            
            # Ищем ближайшие симптомы в БД, исключая указанный
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        se.symptom_id,
                        ds.name as canonical_name,
                        1 - (se.embedding <=> %s::vector) as similarity
                    FROM symptom_embeddings se
                    JOIN diagnosis_symptom ds ON se.symptom_id = ds.id
                    WHERE se.model_name = %s
                        AND ds.name != %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, [embedding_str, self.model_name, symptom_name, top_k + len(exclude)])
                
                results = cursor.fetchall()
            
            # Фильтруем исключённые симптомы
            filtered = []
            for symptom_id, canonical_name, similarity in results:
                if canonical_name not in exclude and similarity >= self.confidence_threshold:
                    filtered.append({
                        'symptom_id': symptom_id,
                        'canonical_name': canonical_name,
                        'confidence': float(similarity)
                    })
                    if len(filtered) >= top_k:
                        break
            
            return filtered
            
        except Exception as e:
            print(f"Ошибка в find_similar: {e}")
            return []
    
    def set_threshold(self, threshold: float):
        """Изменяет порог уверенности."""
        self.confidence_threshold = threshold