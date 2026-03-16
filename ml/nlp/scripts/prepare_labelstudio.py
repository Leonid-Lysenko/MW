"""
Конвертер датасета в формат Label Studio с автоматической BIO-разметкой
"""

import json
import re
from collections import defaultdict
import os

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'generated_phrases_v5.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'labelstudio_ready.json')

# Цвета для разных типов сущностей (для наглядности в консоли)
COLORS = {
    'B-SYMP': '🔴',
    'I-SYMP': '🟠',
    'B-SYMP-NEG': '🔵',
    'I-SYMP-NEG': '🟣',
    'O-NEG': '🟢',
    'O': '⚪'
}

# Карта для исправления ошибок раскладки (английские буквы -> русские)
KEYBOARD_MAP = {
    'f': 'а', ',': 'б', 'd': 'в', 'u': 'г', 'l': 'д', 't': 'е',
    '`': 'ё', ';': 'ж', 'p': 'з', 'b': 'и', 'q': 'й', 'r': 'к',
    'k': 'л', 'v': 'м', 'y': 'н', 'j': 'о', 'g': 'п', 'h': 'р',
    'c': 'с', 'n': 'т', 'e': 'у', 'a': 'ф', '[': 'х', 'w': 'ц',
    'x': 'ч', 'i': 'ш', 'o': 'щ', ']': 'ъ', 's': 'ы', 'm': 'ь',
    "'": 'э', '.': 'ю', 'z': 'я'
}

# Обратная карта (русские буквы -> английские)
RUSSIAN_TO_ENGLISH = {v: k for k, v in KEYBOARD_MAP.items()}

def tokenize(text):
    """
    Разбивает текст на токены (слова и знаки препинания)
    """
    tokens = []
    current_token = ''
    
    for char in text:
        if char.isalnum() or char in '-абвгдеёжзийклмнопрстуфхцчшщъыьэюя' or char in KEYBOARD_MAP.keys():
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ''
            if char.strip():  # если это не пробел
                tokens.append(char)
    if current_token:
        tokens.append(current_token)
    
    return tokens

def levenshtein_distance(s1, s2):
    """
    Расстояние Левенштейна между двумя строками
    """
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

def fix_keyboard_layout(word):
    """
    Исправляет ошибку раскладки (английские буквы -> русские)
    """
    result = []
    for ch in word.lower():
        if ch in KEYBOARD_MAP:
            result.append(KEYBOARD_MAP[ch])
        else:
            result.append(ch)
    return ''.join(result)

def to_english_layout(word):
    """
    Переводит русское слово в английскую раскладку (для обратной проверки)
    """
    result = []
    for ch in word.lower():
        if ch in RUSSIAN_TO_ENGLISH:
            result.append(RUSSIAN_TO_ENGLISH[ch])
        else:
            result.append(ch)
    return ''.join(result)

def normalize_text(text):
    """
    Нормализует текст для сравнения
    """
    # Убираем лишние пробелы
    text = ' '.join(text.split())
    return text

def word_similarity(word1, word2):
    """
    Проверяет, являются ли слова похожими (с учетом опечаток, раскладки и падежей)
    """
    # Точное совпадение
    if word1.lower() == word2.lower():
        return True
    
    # Совпадение после исправления раскладки
    if fix_keyboard_layout(word1) == word2.lower():
        return True
    
    if to_english_layout(word1) == word2.lower():
        return True
    
    # Проверка расстояния Левенштейна (для опечаток)
    max_distance = 2 if len(word1) <= 5 else 3
    if len(word1) > 2 and len(word2) > 2:
        if levenshtein_distance(word1.lower(), word2.lower()) <= max_distance:
            return True
    
    # Проверка на опечатки внутри слова (пропуск букв)
    if len(word1) > 3 and len(word2) > 3:
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            return True
    
    # ПРОВЕРКА ПАДЕЖЕЙ: сравниваем основы слов
    # Убираем окончания для существительных
    word1_stem = word1.lower().rstrip('аяоеёиыуэю')
    word2_stem = word2.lower().rstrip('аяоеёиыуэю')
    if word1_stem == word2_stem and len(word1_stem) > 2:
        return True
    
    return False

def find_fuzzy_matches(text, search_text):
    """
    Находит все вхождения search_text в text с учетом опечаток
    Возвращает список позиций начала и концов
    """
    matches = []
    
    # Нормализуем тексты
    text = normalize_text(text)
    text_words = text.split()
    search_words = search_text.split()
    
    if not search_words:
        return matches
    
    # Ищем последовательность слов
    for i in range(len(text_words) - len(search_words) + 1):
        match = True
        matched_indices = []
        
        for j, search_word in enumerate(search_words):
            candidate = text_words[i + j]
            
            if word_similarity(candidate, search_word):
                matched_indices.append(i + j)
            else:
                match = False
                break
        
        if match and matched_indices:
            # Находим позиции в исходном тексте
            start_pos = text.find(text_words[i])
            
            # Находим конец последнего слова
            last_word = text_words[matched_indices[-1]]
            last_word_pos = text.find(last_word, start_pos)
            end_pos = last_word_pos + len(last_word)
            
            matches.append((start_pos, end_pos))
    
    return matches

def generate_bio_tags(phrase_text, symptoms):
    """
    Генерирует BIO-разметку для фразы на основе списка симптомов
    """
    # Токенизируем текст
    tokens = tokenize(phrase_text)
    tags = ['O'] * len(tokens)
    
    # Отслеживаем размеченные позиции
    marked_spans = []
    
    # Сначала размечаем симптомы (приоритет)
    for symptom in symptoms:
        used_text = symptom['used_text']
        is_negated = symptom['negated']
        
        # Ищем с учетом опечаток
        matches = find_fuzzy_matches(phrase_text, used_text)
        
        for start_pos, end_pos in matches:
            # Проверяем, не пересекается ли с уже размеченным
            overlap = False
            for marked_start, marked_end in marked_spans:
                if not (end_pos <= marked_start or start_pos >= marked_end):
                    overlap = True
                    break
            
            if overlap:
                continue
            
            # Определяем индексы токенов, попадающих в этот диапазон
            token_start_idx = -1
            token_end_idx = -1
            char_pos = 0
            
            for i, token in enumerate(tokens):
                # Ищем токен в тексте
                token_pos = phrase_text.find(token, char_pos)
                if token_pos == -1:
                    continue
                
                token_end = token_pos + len(token)
                
                # Проверяем, попадает ли токен в диапазон
                if token_pos <= end_pos and token_end >= start_pos:
                    if token_start_idx == -1:
                        token_start_idx = i
                    token_end_idx = i
                
                char_pos = token_end
            
            if token_start_idx != -1 and token_end_idx != -1:
                prefix = '-NEG' if is_negated else ''
                tags[token_start_idx] = f'B-SYMP{prefix}'
                for i in range(token_start_idx + 1, token_end_idx + 1):
                    tags[i] = f'I-SYMP{prefix}'
                
                marked_spans.append((start_pos, end_pos))
    
    # Размечаем слова отрицаний
    negation_words = ['нет', 'не', 'без', 'отсутствует', 'кроме', 'ни', 'но']
    for i, token in enumerate(tokens):
        if tags[i] != 'O':
            continue
            
        token_clean = token.lower().strip('.,!?;:')
        
        # Проверяем точное совпадение
        if token_clean in negation_words:
            tags[i] = 'O-NEG'
            continue
        
        # Проверяем с учетом опечаток
        for neg_word in negation_words:
            if word_similarity(token_clean, neg_word):
                tags[i] = 'O-NEG'
                break
    
    return list(zip(tokens, tags))

def visualize_tags(phrase_text, token_tags):
    """
    Визуализирует размеченный текст для проверки
    """
    result = []
    for token, tag in token_tags:
        color = COLORS.get(tag, '⚫')
        result.append(f"{color} {token} [{tag}]")
    
    return ' '.join(result)

def main():
    print("=" * 60)
    print("КОНВЕРТЕР ДАТАСЕТА В ФОРМАТ LABEL STUDIO")
    print("=" * 60)
    
    # Загружаем сгенерированные фразы
    try:
        with open(INPUT_PATH, 'r', encoding='utf-8') as f:
            phrases = json.load(f)
        print(f"Загружено {len(phrases)} фраз из {INPUT_PATH}")
    except FileNotFoundError:
        print(f"Файл не найден: {INPUT_PATH}")
        return
    
    # Конвертируем в формат Label Studio
    labelstudio_tasks = []
    total_entities = 0
    tag_counts = defaultdict(int)
    phrases_with_issues = []
    
    for i, phrase in enumerate(phrases):
        # Генерируем BIO-теги
        token_tags = generate_bio_tags(phrase['phrase_text'], phrase['symptoms'])
        
        # Считаем сущности
        entities_count = 0
        current_entity = None
        
        for token, tag in token_tags:
            if tag != 'O':
                entities_count += 1
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
        
        # Добавляем разметку в формате Label Studio
        char_pos = 0
        for token, tag in token_tags:
            if tag != 'O':  # Пропускаем O, размечаем только сущности
                # Находим точную позицию токена
                start = phrase['phrase_text'].find(token, char_pos)
                if start != -1:
                    end = start + len(token)
                    
                    # Убеждаемся, что мы не вышли за границы
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
                        char_pos = end
        
        # Проверяем, все ли симптомы размечены
        if entities_count == 0 and len(phrase['symptoms']) > 0:
            phrases_with_issues.append({
                'id': i + 1,
                'text': phrase['phrase_text'][:50] + '...',
                'symptoms': len(phrase['symptoms'])
            })
        
        labelstudio_tasks.append(task)
        
        # Показываем пример первых 3 фраз
        if i < 3:
            print(f"\nПример {i+1}:")
            print(f"Текст: {phrase['phrase_text']}")
            print(f"Разметка: {visualize_tags(phrase['phrase_text'], token_tags)}")
    
    # Сохраняем в формате, понятном Label Studio
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_tasks, f, ensure_ascii=False, indent=2)
    
    print(f"\nСохранено {len(labelstudio_tasks)} задач в {OUTPUT_PATH}")
    
    # Статистика
    print("\n" + "=" * 60)
    print("СТАТИСТИКА РАЗМЕТКИ")
    print("=" * 60)
    print(f"Всего фраз: {len(phrases)}")
    print(f"Всего размеченных сущностей: {total_entities}")
    print(f"В среднем: {total_entities/len(phrases):.1f} на фразу")
    
    print("\n   Распределение по типам:")
    for tag, count in sorted(tag_counts.items()):
        color = COLORS.get(tag, '⚫')
        percentage = (count / total_entities) * 100 if total_entities > 0 else 0
        print(f"      {color} {tag}: {count} ({percentage:.1f}%)")
    
    # Проблемные фразы
    if phrases_with_issues:
        print(f"\nНайдено {len(phrases_with_issues)} фраз, где не удалось найти симптомы:")
        for issue in phrases_with_issues[:5]:
            print(f"Фраза {issue['id']}: {issue['text']} (ожидалось {issue['symptoms']} симптомов)")
    else:
        print(f"\nВсе фразы размечены корректно!")
    
    print(f"\nФайл готов для импорта в Label Studio: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()