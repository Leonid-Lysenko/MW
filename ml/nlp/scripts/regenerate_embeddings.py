# regenerate_embeddings.py
import os
import sys
import django
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

# Настройка Django
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
backend_dir = os.path.join(project_root, 'backend')

sys.path.append(project_root)
sys.path.append(backend_dir)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_site.settings')
django.setup()

from django.db import connection

# ===== КОНСТАНТЫ =====
MODEL_NAME = 'intfloat/multilingual-e5-large'  # E5-large

def get_all_symptoms():
    """Получает все симптомы из БД."""
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, name FROM diagnosis_symptom ORDER BY id")
        return cursor.fetchall()

def clear_embeddings():
    """Удаляет старые эмбеддинги."""
    print(" Удаление старых эмбеддингов...")
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM symptom_embeddings WHERE model_name = 'e5-large'")
        connection.commit()
    print(" Старые эмбеддинги удалены")

def generate_embeddings(symptoms):
    """Генерирует новые эмбеддинги для всех симптомов."""
    print(" Загрузка модели E5-large...")
    model = SentenceTransformer(MODEL_NAME)
    
    print(f" Генерация эмбеддингов для {len(symptoms)} симптомов...")
    
    embeddings = []
    for i, (symptom_id, name) in enumerate(symptoms):
        # E5 требует префикс "query:" для поиска
        text = f"query: {name}"
        embedding = model.encode(text, normalize_embeddings=True)
        embeddings.append((symptom_id, embedding))
        
        if (i + 1) % 50 == 0:
            print(f"   Обработано {i + 1}/{len(symptoms)}")
    
    return embeddings

def save_embeddings(embeddings):
    """Сохраняет эмбеддинги в БД."""
    print("💾 Сохранение эмбеддингов в БД...")
    
    with connection.cursor() as cursor:
        for symptom_id, embedding in embeddings:
            embedding_str = '[' + ','.join(f"{x:.8f}" for x in embedding.tolist()) + ']'
            cursor.execute("""
                INSERT INTO symptom_embeddings (symptom_id, model_name, embedding)
                VALUES (%s, %s, %s::vector)
            """, [symptom_id, 'e5-large', embedding_str])
        
        connection.commit()
    
    print(f" Сохранено {len(embeddings)} эмбеддингов")

def main():
    print("="*60)
    print("ПЕРЕГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ДЛЯ СЕМАНТИЧЕСКОГО ПОИСКА")
    print("="*60)
    
    # 1. Получаем симптомы
    symptoms = get_all_symptoms()
    print(f" Найдено симптомов в БД: {len(symptoms)}")
    
    # 2. Удаляем старые эмбеддинги
    clear_embeddings()
    
    # 3. Генерируем новые
    embeddings = generate_embeddings(symptoms)
    
    # 4. Сохраняем
    save_embeddings(embeddings)
    
    print("\n Готово! Семантический поиск обновлён.")
    print(" Всего эмбеддингов: 304")

if __name__ == "__main__":
    main()