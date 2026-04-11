# backend/diagnosis/symptoms_data.py
"""
Список симптомов, загруженный из ml/nlp/data/synonyms/synonyms_updated.json
"""

import json
import os

def load_symptoms():
    """Загружает список симптомов из файла синонимов."""
    try:
        # Путь к файлу синонимов
        current_dir = os.path.dirname(os.path.abspath(__file__))
        synonyms_path = os.path.join(
            current_dir, '..', '..', 'ml', 'nlp', 'data', 'synonyms', 'synonyms_updated.json'
        )
        synonyms_path = os.path.normpath(synonyms_path)
        
        with open(synonyms_path, 'r', encoding='utf-8') as f:
            synonyms_data = json.load(f)
        
        # Симптомы — это ключи словаря
        return list(synonyms_data.keys())
    except Exception as e:
        print(f"Ошибка загрузки симптомов: {e}")
        return []

# Загружаем симптомы при старте
SYMPTOMS_LIST = load_symptoms()