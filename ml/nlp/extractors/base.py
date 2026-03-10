# ml/nlp/extractors/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SymptomExtractor(ABC):
    """
    Абстрактный базовый класс для всех экстракторов симптомов.
    Все модули (rule-based, семантический поиск, NER, гибридный)
    должны наследоваться от этого класса и реализовывать метод extract.
    """
    
    @abstractmethod
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает симптомы из текста.
        
        Args:
            text: Входной текст на русском языке (до 500 символов)
            
        Returns:
            Список словарей с ключами:
            - symptom_id: int (идентификатор симптома из БД)
            - canonical_name: str (каноническое название)
            - status: str ('present' или 'absent')
            - confidence: float (уверенность, 0.0-1.0)
            - matched_text: str (фрагмент текста, по которому нашли)
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Общая предобработка текста для всех методов.
        """
        if not text:
            return ""
        # Приводим к нижнему регистру
        text = text.lower()
        # Заменяем знаки препинания на пробелы
        import re
        text = re.sub(r'[^\w\s]', ' ', text)
        # Убираем лишние пробелы
        text = ' '.join(text.split())
        return text