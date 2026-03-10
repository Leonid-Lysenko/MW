# ml/nlp/utils/text_processing.py
import re
import pymorphy3
from typing import List, Tuple, Optional

# Инициализируем морфологический анализатор
morph = pymorphy3.MorphAnalyzer()

# Полный список маркеров отрицания
NEGATION_WORDS = {
    # Основные
    'нет', 'не', 'без', 'отсутствует', 'кроме',
    # Глаголы с семантикой отрицания
    'перестал', 'перестала', 'перестало', 'перестали',
    'прошел', 'прошла', 'прошло', 'прошли',
    'исчез', 'исчезла', 'исчезло', 'исчезли',
    'пропал', 'пропала', 'пропало', 'пропали',
    'прекратился', 'прекратилась', 'прекратилось', 'прекратились',
}

# Составные паттерны отрицания (регулярные выражения)
NEGATION_PATTERNS = [
    # Основные
    (r'\bнет\s+', 'before'),           # "нет кашля"
    (r'\bне\s+', 'before'),            # "не болит"
    (r'\bбез\s+', 'before'),           # "без температуры"
    (r'\bкроме\s+', 'before'),         # "кроме кашля"
    
    # "отсутствует" может быть после
    (r'\s+отсутствует\b', 'after'),    # "кашель отсутствует"
    (r'\s+отсутствуют\b', 'after'),    # "симптомы отсутствуют"
    
    # Глаголы после симптома
    (r'\s+перестал\b', 'after'),       # "кашель перестал"
    (r'\s+прошел\b', 'after'),          # "кашель прошел"
    (r'\s+исчез\b', 'after'),           # "головная боль исчезла"
    (r'\s+пропал\b', 'after'),          # "голос пропал"
    (r'\s+прекратился\b', 'after'),     # "насморк прекратился"
    
    # "нет" в конце
    (r'\s+нет\b', 'after'),             # "кашля нет"
    
    # "ни" конструкция
    (r'\bни\s+', 'before'),             # "ни кашля"
]

def tokenize(text: str) -> List[str]:
    """Разбивает текст на слова (токены)."""
    return re.findall(r'\b\w+\b', text.lower())

def find_negations(
    text: str, 
    symptom_start: int, 
    symptom_end: int, 
    window_size_words: int = 3
) -> bool:
    """
    Проверяет, есть ли отрицание в окрестности симптома.
    
    Args:
        text: исходный текст (оригинальный, для сохранения регистра)
        symptom_start: позиция начала симптома в символах
        symptom_end: позиция конца симптома в символах
        window_size_words: размер окна в словах для поиска отрицания
        
    Returns:
        True если найдено отрицание, иначе False
    """
    # Приводим к нижнему регистру для поиска
    text_lower = text.lower()
    
    # Разбиваем на слова для работы с окном
    words = tokenize(text_lower)
    
    # Находим индекс слова, в которое попадает симптом
    char_pos = 0
    symptom_word_idx = -1
    word_positions = []
    
    for i, word in enumerate(words):
        start = text_lower.find(word, char_pos)
        if start == -1:
            continue
        end = start + len(word)
        word_positions.append((start, end, word))
        
        # Проверяем, перекрывается ли слово с симптомом
        if (start <= symptom_start < end) or (start < symptom_end <= end) or (symptom_start <= start and symptom_end >= end):
            symptom_word_idx = i
        
        char_pos = end
    
    if symptom_word_idx == -1:
        return False  # симптом не найден в словах (странно)
    
    # Формируем контекст: N слов слева и N слов справа
    context_start = max(0, symptom_word_idx - window_size_words)
    context_end = min(len(words), symptom_word_idx + window_size_words + 1)
    
    context_words = words[context_start:context_end]
    context_str = ' '.join(context_words)
    
    # 1. Проверяем отдельные слова отрицания
    for word in context_words:
        if word in NEGATION_WORDS:
            return True
    
    # 2. Проверяем паттерны
    for pattern, position in NEGATION_PATTERNS:
        if re.search(pattern, ' ' + context_str + ' '):
            return True
    
    # 3. Специальная проверка для "ни X ни Y"
    if 'ни' in context_words:
        # Проверяем, что "ни" относится к нашему симптому
        # (упрощенно: если есть "ни" и наш симптом среди слов с "ни")
        return True
    
    # 4. Проверка для "кроме X" (отрицание относится к X)
    # Если есть "кроме" и наш симптом после него
    for i, word in enumerate(context_words):
        if word == 'кроме' and i < len(context_words) - 1:
            # Проверяем, что следующий симптом (или один из) - наш
            return True
    
    return False

def normalize_text(text: str) -> str:
    """
    Нормализует текст для поиска: приводит к нижнему регистру,
    убирает пунктуацию, приводит слова к начальной форме.
    """
    if not text:
        return ""
    
    # Приводим к нижнему регистру
    text = text.lower()
    
    # Убираем пунктуацию
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Разбиваем на слова
    words = text.split()
    
    # Приводим каждое слово к начальной форме (нормальной форме)
    normalized_words = []
    for word in words:
        try:
            # Пытаемся привести к нормальной форме
            parsed = morph.parse(word)[0]
            normal_form = parsed.normal_form
            normalized_words.append(normal_form)
        except:
            # Если ошибка, оставляем как есть
            normalized_words.append(word)
    
    # Собираем обратно
    return ' '.join(normalized_words)

def get_word_context(text: str, position: int, window: int = 3) -> str:
    """
    Возвращает контекст из N слов вокруг указанной позиции.
    Полезно для отладки.
    """
    words = tokenize(text)
    char_pos = 0
    target_idx = -1
    
    for i, word in enumerate(words):
        start = text.lower().find(word, char_pos)
        if start == -1:
            continue
        end = start + len(word)
        if start <= position < end:
            target_idx = i
            break
        char_pos = end
    
    if target_idx == -1:
        return ""
    
    start_idx = max(0, target_idx - window)
    end_idx = min(len(words), target_idx + window + 1)
    
    return ' '.join(words[start_idx:end_idx])