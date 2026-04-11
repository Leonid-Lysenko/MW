# ml/nlp/extractors/semantic_search.py
import numpy as np
from typing import List, Dict, Any, Optional, Set
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from django.db import connection
from .base import SymptomExtractor
import torch


class SemanticSearchExtractor(SymptomExtractor):
    """
    Экстрактор симптомов на основе семантического поиска.
    Использует эмбеддинги и pgvector для поиска ближайших симптомов.
    """
    
    def __init__(self, model_name: str = 'intfloat/multilingual-e5-small', confidence_threshold: float = 0.6):
        super().__init__()
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
        
    def _load_model(self):
        """Загружает соответствующую модель."""
        if self.model_name == 'intfloat/multilingual-e5-small':
            return SentenceTransformer('intfloat/multilingual-e5-small')
        elif self.model_name == 'e5-large':
            return SentenceTransformer('intfloat/multilingual-e5-large')
        elif self.model_name == 'rubioroberta':
            self.tokenizer = AutoTokenizer.from_pretrained('alexyalunin/RuBioRoBERTa')
            model = AutoModel.from_pretrained('alexyalunin/RuBioRoBERTa')
            model.eval()
            return model
        else:
            raise ValueError(f"Неизвестная модель: {self.model_name}")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Генерирует эмбеддинг для текста."""
        if self.model_name in ['intfloat/multilingual-e5-small', 'e5-large']:
            text = f"query: {text}"
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        
        elif self.model_name == 'rubioroberta':
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embedding = embedding.squeeze().numpy()
                embedding = embedding / np.linalg.norm(embedding)
            return embedding.tolist()
    
    def extract(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not text:
            return []
        
        query_embedding = self._get_embedding(text)
        embedding_str = '[' + ','.join(f"{x:.8f}" for x in query_embedding) + ']'
        
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    se.symptom_id,
                    ds.name as canonical_name,
                    1 - (se.embedding <=> %s::vector) as similarity
                FROM symptom_embeddings_small se
                JOIN diagnosis_symptom ds ON se.symptom_id = ds.id
                WHERE se.model_name = %s
                ORDER BY similarity DESC
                LIMIT %s
            """, [embedding_str, self.model_name, top_k])
            
            results = cursor.fetchall()
        
        symptoms = []
        for symptom_id, canonical_name, similarity in results:
            if similarity >= self.confidence_threshold:
                symptoms.append({
                    'symptom_id': symptom_id,
                    'canonical_name': canonical_name,
                    'status': 'present',
                    'confidence': float(similarity),
                    'matched_text': text
                })
        
        return symptoms
    
    def find_similar(self, symptom_name: str, exclude: Set[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        if exclude is None:
            exclude = set()
        
        if not symptom_name:
            return []
        
        try:
            embedding = self._get_embedding(symptom_name)
            embedding_str = '[' + ','.join(f"{x:.8f}" for x in embedding) + ']'
            
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        se.symptom_id,
                        ds.name as canonical_name,
                        1 - (se.embedding <=> %s::vector) as similarity
                    FROM symptom_embeddings_small se
                    JOIN diagnosis_symptom ds ON se.symptom_id = ds.id
                    WHERE se.model_name = %s
                        AND ds.name != %s
                    ORDER BY similarity DESC
                    LIMIT %s
                """, [embedding_str, self.model_name, symptom_name, top_k + len(exclude)])
                
                results = cursor.fetchall()
            
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
            return []
    
    def set_threshold(self, threshold: float):
        self.confidence_threshold = threshold