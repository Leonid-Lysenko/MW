"""
Конвертер датасета в формат Label Studio с автоматической BIO-разметкой
"""

import json
import re
from collections import defaultdict
import os

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'merged_dataset.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'labelstudio_ready.json')

# Цвета для разных типов сущностей
COLORS = {
    'B-SYMP': '🔴',
    'I-SYMP': '🟠',
    'B-SYMP-NEG': '🔵',
    'I-SYMP-NEG': '🟣',
    'O': '⚪'
}

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def levenshtein_distance(s1, s2):
    """Расстояние Левенштейна"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def word_similarity(word1, word2):
    """Проверяет похожесть слов (0.0 - 1.0)"""
    w1 = word1.lower()
    w2 = word2.lower()
    
    if w1 == w2:
        return 1.0
    
    # Расстояние Левенштейна
    max_len = max(len(w1), len(w2))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(w1, w2)
    similarity = 1 - (dist / max_len)
    
    return similarity

def preprocess_text(phrase_text):
    """
    Преобразует текст в список слов.
    Сохраняет дефисы внутри слов (например, "что-то" остаётся одним словом).
    """
    # Заменяем все знаки препинания, кроме дефиса, на пробелы
    text = re.sub(r'[^\w\s\-]', ' ', phrase_text.lower())
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    # Разбиваем на слова
    words = text.split()
    return words

def tokenize_with_positions(phrase_text):
    """
    Разбивает текст на токены с сохранением позиций
    Возвращает список токенов и список позиций (start, end)
    """
    tokens = []
    positions = []
    
    i = 0
    while i < len(phrase_text):
        # Пропускаем пробелы
        if phrase_text[i].isspace():
            i += 1
            continue
        
        start = i
        
        # Определяем тип символа
        if phrase_text[i].isalnum() or phrase_text[i] in '-абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
            # Буква, цифра или дефис
            while i < len(phrase_text) and (phrase_text[i].isalnum() or phrase_text[i] in '-абвгдеёжзийклмнопрстуфхцчшщъыьэюя'):
                i += 1
        else:
            # Знак препинания
            i += 1
        
        if start < i:
            token = phrase_text[start:i]
            tokens.append(token)
            positions.append((start, i))
    
    return tokens, positions

def find_word_sequence(search_words, text_words, used_indices, start_from=0):
    """
    Ищет последовательность search_words в тексте в прямом порядке
    """
    n = len(search_words)
    if n == 0:
        return []
    
    for i in range(start_from, len(text_words) - n + 1):
        # Проверяем, не заняты ли позиции
        skip = False
        for j in range(n):
            if i + j in used_indices:
                skip = True
                break
        if skip:
            continue
        
        # Проверяем совпадение
        match = True
        for j, search_word in enumerate(search_words):
            if word_similarity(search_word, text_words[i + j]) < 0.5:
                match = False
                break
        
        if match:
            return list(range(i, i + n))
    
    return []

def process_symptom(symptom, text_words, used_indices):
    """
    Обрабатывает один симптом, возвращает список меток для слов
    """
    used_text = symptom['used_text']
    is_negated = symptom['negated']
    
    # Очищаем used_text от знаков препинания
    used_text_clean = re.sub(r'[^\w\s\-]', ' ', used_text.lower())
    used_text_clean = ' '.join(used_text_clean.split())
    search_words = used_text_clean.split()
    
    n_search = len(search_words)
    if n_search == 0:
        return []
    
    # Ищем последовательность слов, начиная с 0
    # Но пропускаем уже использованные индексы
    for i in range(len(text_words) - n_search + 1):
        # Проверяем, не заняты ли позиции
        skip = False
        for j in range(n_search):
            if i + j in used_indices:
                skip = True
                break
        if skip:
            continue
        
        # Проверяем совпадение
        match = True
        for j, search_word in enumerate(search_words):
            if word_similarity(search_word, text_words[i + j]) < 0.5:
                match = False
                break
        
        if match:
            result = []
            for j, idx in enumerate(range(i, i + n_search)):
                if j == 0:
                    label = 'B-SYMP' + ('-NEG' if is_negated else '')
                else:
                    label = 'I-SYMP' + ('-NEG' if is_negated else '')
                result.append((idx, label, text_words[idx]))
            return result
    
    return []

def get_token_tags_from_words(phrase_text, text_words, word_tags):
    """
    Преобразует метки слов в метки токенов для Label Studio
    """
    # Токенизируем оригинальный текст
    tokens, token_positions = tokenize_with_positions(phrase_text)
    
    # Создаём список слов из текста с их позициями
    text_word_positions = []
    pos = 0
    for w in text_words:
        start = phrase_text.lower().find(w, pos)
        if start != -1:
            text_word_positions.append((w, start, start + len(w)))
            pos = start + len(w)
    
    # Сопоставляем токены с метками слов
    token_tags = ['O'] * len(tokens)
    
    token_idx = 0
    word_idx = 0
    
    while token_idx < len(tokens) and word_idx < len(text_words):
        token = tokens[token_idx].lower().strip('.,!?;:()[]')
        word, w_start, w_end = text_word_positions[word_idx]
        
        token_start, token_end = token_positions[token_idx]
        
        # Проверяем, принадлежит ли токен этому слову
        if token_start >= w_start and token_end <= w_end:
            token_tags[token_idx] = word_tags[word_idx]
            token_idx += 1
            # Если токен закончился, переходим к следующему слову
            if token_end >= w_end:
                word_idx += 1
        else:
            token_idx += 1
    
    return list(zip(tokens, token_tags))

def generate_bio_tags(phrase_text, symptoms):
    """
    Генерирует BIO-разметку с отсечением обработанных слов
    """
    # Преобразуем текст в список слов
    text_words = preprocess_text(phrase_text)
    
    # Создаём массив меток (изначально все O)
    word_tags = ['O'] * len(text_words)
    
    # Отслеживаем использованные индексы
    used_indices = set()
    
    # Сортируем симптомы по длине used_text (сначала самые длинные)
    symptoms_sorted = sorted(symptoms, key=lambda x: len(x['used_text'].split()), reverse=True)
    
    # Обрабатываем симптомы по очереди
    for symptom in symptoms_sorted:
        result = process_symptom(symptom, text_words, used_indices)
        
        if result:
            for idx, label, word in result:
                word_tags[idx] = label
                used_indices.add(idx)
    
    # Преобразуем в токены для Label Studio
    token_tags = get_token_tags_from_words(phrase_text, text_words, word_tags)
    
    return token_tags

def visualize_tags(phrase_text, token_tags):
    """Визуализирует размеченный текст"""
    result = []
    for token, tag in token_tags:
        color = COLORS.get(tag, '⚪')
        result.append(f"{color} {token} [{tag}]")
    return ' '.join(result)

def main():
    print("=" * 60)
    print("КОНВЕРТЕР ДАТАСЕТА В ФОРМАТ LABEL STUDIO v9.0")
    print("=" * 60)
    
    # Загружаем датасет
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            phrases = json.load(f)
        print(f" Загружено {len(phrases)} фраз из {INPUT_PATH}")
    except FileNotFoundError:
        print(f" Файл не найден: {INPUT_PATH}")
        return
    
    labelstudio_tasks = []
    total_entities = 0
    tag_counts = defaultdict(int)
    phrases_with_issues = []
    
    for i, phrase in enumerate(phrases):
        # Генерируем BIO-теги
        token_tags = generate_bio_tags(phrase['phrase_text'], phrase['symptoms'])
        
        # Считаем сущности
        entities_count = sum(1 for _, tag in token_tags if tag != 'O')
        for _, tag in token_tags:
            if tag != 'O':
                tag_counts[tag] += 1
                total_entities += 1
        
        # Формируем задачу для Label Studio
        task = {
            'id': i + 1,
            'data': {
                'phrase_text': phrase['phrase_text'],
                'phrase_id': phrase.get('phrase_id', i + 1)
            },
            'annotations': [{
                'result': [],
                'ground_truth': True
            }]
        }
        
        # Добавляем разметку
        for token, tag in token_tags:
            if tag != 'O':
                start = phrase['phrase_text'].find(token)
                if start != -1:
                    end = start + len(token)
                    if end <= len(phrase['phrase_text']):
                        task['annotations'][0]['result'].append({
                            'id': f"entity_{len(task['annotations'][0]['result'])}",
                            'value': {
                                'start': start,
                                'end': end,
                                'text': token,
                                'labels': [tag]
                            },
                            'from_name': 'label',
                            'to_name': 'text',
                            'type': 'labels'
                        })
        
        # Проверяем, все ли симптомы размечены
        if entities_count == 0 and len(phrase['symptoms']) > 0:
            phrases_with_issues.append({
                'id': i + 1,
                'text': phrase['phrase_text'][:50] + '...',
                'symptoms': len(phrase['symptoms'])
            })
        
        labelstudio_tasks.append(task)
        
        # Показываем примеры
        if i < 3:
            print(f"\n Пример {i+1}:")
            print(f"   Текст: {phrase['phrase_text']}")
            print(f"   Разметка: {visualize_tags(phrase['phrase_text'], token_tags)}")
    
    # Сохраняем
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_tasks, f, ensure_ascii=False, indent=2)
    
    print(f"\n Сохранено {len(labelstudio_tasks)} задач в {OUTPUT_PATH}")
    
    # Статистика
    print("\n" + "=" * 60)
    print(" СТАТИСТИКА РАЗМЕТКИ")
    print("=" * 60)
    print(f"   Всего фраз: {len(phrases)}")
    print(f"   Всего размеченных сущностей: {total_entities}")
    print(f"   В среднем: {total_entities/len(phrases):.1f} на фразу")
    
    print("\n   Распределение по типам:")
    for tag, count in sorted(tag_counts.items()):
        color = COLORS.get(tag, '⚪')
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        print(f"      {color} {tag}: {count} ({percentage:.1f}%)")
    
    if phrases_with_issues:
        print(f"\n Найдено {len(phrases_with_issues)} фраз, где не удалось найти симптомы:")
        for issue in phrases_with_issues[:5]:
            print(f"   Фраза {issue['id']}: {issue['text']} (ожидалось {issue['symptoms']} симптомов)")
    else:
        print(f"\n Все фразы размечены корректно!")
    
    print(f"\n Файл готов для импорта в Label Studio: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()