"""
Генератор датасета для извлечения симптомов из текста.
"""

import json
import random
import psycopg2
from psycopg2.extras import DictCursor
from typing import List, Dict, Tuple
import os
from collections import Counter, defaultdict
import re

# ====================== НАСТРОЙКИ ======================
DB_CONFIG = {
    'dbname': 'diagnostics_db',
    'user': 'postgres',
    'password': 'medic1312',
    'host': 'localhost',
    'port': '5432'
}

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYNONYMS_PATH = os.path.join(BASE_DIR, '..', 'data', 'synonyms', 'synonyms.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'generated_phrases_v5.json')

# Параметры генерации
RANDOM_SEED = 42
N_PHRASES_TOTAL = 2000
NEGATION_RATIO = 0.10          # 10% фраз с отрицаниями
TYPO_RATIO = 0.15               # 15% фраз с опечатками
KEYBOARD_ERROR_RATIO = 0.05     # 5% фраз с ошибкой раскладки
SYNONYM_PROBABILITY = 0.7       # 70% симптомов заменяются на синонимы

# ====================== БАЛАНС КЛАССОВ ======================
# Вероятности количества симптомов в фразе
SYMPTOM_COUNT_PROBS = {
    1: 0.3,   # 30% фраз с 1 симптомом
    2: 0.35,  # 35% фраз с 2 симптомами
    3: 0.25,  # 25% фраз с 3 симптомами
    4: 0.1    # 10% фраз с 4 симптомами
}

# Длина шума в токенах
SHORT_NOISE_LEN = (3, 6)      # короткий шум: 3-6 токенов
MEDIUM_NOISE_LEN = (7, 12)    # средний шум: 7-12 токенов
LONG_NOISE_LEN = (13, 20)     # длинный шум: 13-20 токенов

NOISE_LENGTH_PROBS = {
    'short': 0.3,   # 30% фраз с коротким шумом
    'medium': 0.5,  # 50% фраз со средним шумом
    'long': 0.2     # 20% фраз с длинным шумом
}

# ====================== ЗАГРУЗКА СВЯЗЕЙ СИМПТОМОВ ======================

def load_disease_symptom_correlations():
    """
    Загружает связи между симптомами из базы данных заболеваний.
    Возвращает словарь: для каждого симптома - список симптомов,
    которые часто с ним встречаются.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)
        
        # Получаем все заболевания и их симптомы
        cur.execute("""
            SELECT d.id, array_agg(s.id) as symptom_ids
            FROM diagnosis_disease d
            JOIN diagnosis_disease_symptoms ds ON d.id = ds.disease_id
            JOIN diagnosis_symptom s ON ds.symptom_id = s.id
            GROUP BY d.id
        """)
        
        disease_symptoms = cur.fetchall()
        cur.close()
        conn.close()
        
        # Строим граф совместной встречаемости симптомов
        cooccurrence = defaultdict(int)
        symptom_pairs = []
        
        for row in disease_symptoms:
            symptoms = row['symptom_ids']
            # Для каждого заболевания считаем все пары симптомов
            for i in range(len(symptoms)):
                for j in range(i+1, len(symptoms)):
                    pair = tuple(sorted([symptoms[i], symptoms[j]]))
                    cooccurrence[pair] += 1
                    symptom_pairs.append(pair)
        
        print(f"Загружено {len(disease_symptoms)} заболеваний")
        print(f"Найдено {len(cooccurrence)} пар симптомов")
        
        return cooccurrence, symptom_pairs
    except Exception as e:
        print(f"Не удалось загрузить корреляции: {e}")
        return {}, []

# ====================== СЛОВАРИ ДЛЯ ШУМА ======================

# Вводные слова и фразы
INTRO_WORDS = [
    "последнее время", "последние пару дней", "сегодня утром", "вчера вечером",
    "уже неделю", "после простуды", "на фоне стресса", "к вечеру",
    "утром", "днем", "ночью", "после еды", "на голодный желудок",
    "при ходьбе", "в покое", "при нагрузке", "после сна",
    "за последний месяц", "после тренировки", "во время отдыха",
    "с утра", "вечером", "ночью", "днём", "после работы",
    "в последнее время", "раньше", "теперь", "сейчас"
]

# Слова-интенсификаторы
INTENSIFIERS = [
    "сильно", "очень", "довольно", "весьма", "невыносимо",
    "постоянно", "периодически", "иногда", "часто", "редко",
    "умеренно", "слабо", "резко", "внезапно", "внезапная",
    "сильная", "острая", "тупая", "ноющая", "пульсирующая",
    "давящая", "колющая", "режущая", "жгучая", "сжимающая",
    "нестерпимо", "терпимо", "легко", "тяжело"
]

# Союзы и связки
CONJUNCTIONS = ["и", "а", "но", "при этом", "также", "еще", "к тому же",
                "однако", "зато", "хотя", "причем", "вместе с тем",
                "кроме того", "более того", "вдобавок"]

# Слова для временных маркеров
TIME_MARKERS = [
    "уже", "еще", "все еще", "по-прежнему", "раньше", "сейчас",
    "сегодня", "вчера", "позавчера", "завтра", "на днях",
    "неделю назад", "месяц назад", "в прошлом году", "вчера ночью",
    "сегодня утром", "сегодня днем", "сегодня вечером",
    "недавно", "давно", "в последнее время", "последнее время"
]

# "Пустые" фразы (не симптомы) - ДЛИННЫЕ
NOISE_PHRASES = [
    "не могу понять что со мной происходит",
    "подскажите пожалуйста к какому врачу обратиться",
    "стоит ли идти в больницу или само пройдет",
    "очень переживаю по этому поводу",
    "чувствую себя просто ужасно в последнее время",
    "состояние неважное уже который день",
    "беспокоит уже давно и ничего не помогает",
    "может это из-за стресса на работе",
    "раньше такого никогда не было",
    "думал само пройдет но нет",
    "устал от этого состояния очень сильно",
    "никак не могу понять причину",
    "что это может быть подскажите",
    "есть у кого-нибудь такое было",
    "как с этим бороться вообще",
    "уже сил нет терпеть это все",
    "к врачу записался но еще не скоро",
    "может народные средства помогут",
    "интересно это опасно или нет",
    "боюсь что это что-то серьезное",
    "в интернете прочитал что это может быть опасно",
    "врач сказал что ничего страшного",
    "терапевт назначил лечение",
    "анализы вроде нормальные",
    "температуры вроде нет",
    "давление в норме"
]

# Слова, которые могут разрывать составные симптомы
INTERNAL_NOISE = [
    "очень", "довольно", "весьма", "невыносимо",
    "постоянная", "сильная", "острая", "тупая",
    "ноющая", "пульсирующая", "давящая", "режущая",
    "жгучая", "колющая", "сжимающая", "распирающая",
    "небольшая", "умеренная", "легкая"
]

# Дополнительные шумовые слова
EXTRA_NOISE_WORDS = [
    "вот", "это", "такое", "что-то", "как-то", "прямо",
    "типа", "вроде", "кажется", "похоже", "наверное",
    "возможно", "конечно", "естественно", "очевидно",
    "вообще", "просто", "прямо", "совсем", "абсолютно",
    "почему-то", "зачем-то", "где-то", "когда-то",
    "ладно", "хорошо", "плохо", "нормально"
]

# ====================== ОТРИЦАНИЯ ======================
NEGATION_WORDS = [
    "не", "нет", "без", "отсутствует", "кроме", "ни"
]

NEGATION_PHRASES = [
    "нет {s}",
    "{s} нет",
    "не {s}",
    "без {s}",
    "{s} отсутствует",
    "отсутствует {s}",
    "кроме {s} ничего",
    "ни {s}",
    "ни {s} ни"
]

# ====================== ШАБЛОНЫ ======================
TEMPLATES = [
    # ===== ШАБЛОНЫ С БОЛЬШИМ КОЛИЧЕСТВОМ ШУМА =====
    {
        'template': '{intro} {s1} {intensifier}. {noise} {time} {s2}.',
        'noise_slots': ['intro', 'intensifier', 'noise', 'time'],
        'min_symptoms': 2
    },
    {
        'template': '{noise} {time} {s1}, {conjunction} {s2} {intensifier}. {extra}',
        'noise_slots': ['noise', 'time', 'conjunction', 'intensifier', 'extra'],
        'min_symptoms': 2
    },
    {
        'template': '{intro} {s1} {intensifier}, {conjunction} {time} {s2} {intensifier}. {noise}',
        'noise_slots': ['intro', 'intensifier', 'conjunction', 'time', 'intensifier', 'noise'],
        'min_symptoms': 2
    },
    {
        'template': '{extra} {s1} {intensifier}. {noise} {conjunction} {time} {s2}.',
        'noise_slots': ['extra', 'intensifier', 'noise', 'conjunction', 'time'],
        'min_symptoms': 2
    },
    {
        'template': '{intro} {s1}. {noise} {conjunction} {intro} {s2} {intensifier}.',
        'noise_slots': ['intro', 'noise', 'conjunction', 'intro', 'intensifier'],
        'min_symptoms': 2
    },
    {
        'template': '{noise} {s1} {intensifier} {conjunction} {s2}. {time} {extra}',
        'noise_slots': ['noise', 'intensifier', 'conjunction', 'time', 'extra'],
        'min_symptoms': 2
    },
    {
        'template': '{time} {s1} {intensifier}. {noise} {conjunction} {s2} {intensifier}. {intro}',
        'noise_slots': ['time', 'intensifier', 'noise', 'conjunction', 'intensifier', 'intro'],
        'min_symptoms': 2
    },
    
    # ===== ШАБЛОНЫ С ОДНИМ СИМПТОМОМ (МНОГО ШУМА) =====
    {
        'template': '{noise} {intro} {s1} {intensifier}. {extra}',
        'noise_slots': ['noise', 'intro', 'intensifier', 'extra'],
        'min_symptoms': 1
    },
    {
        'template': '{intro} {s1} {intensifier} {time}. {noise}',
        'noise_slots': ['intro', 'intensifier', 'time', 'noise'],
        'min_symptoms': 1
    },
    {
        'template': '{time} {s1}. {noise} {conjunction} {extra}',
        'noise_slots': ['time', 'noise', 'conjunction', 'extra'],
        'min_symptoms': 1
    },
    {
        'template': '{intro} {s1} {intensifier}. {extra} {time}',
        'noise_slots': ['intro', 'intensifier', 'extra', 'time'],
        'min_symptoms': 1
    },
    
    # ===== ШАБЛОНЫ С ТРЕМЯ СИМПТОМАМИ =====
    {
        'template': '{intro} {s1}, {conjunction} {s2} {intensifier} {conjunction} {s3}. {noise}',
        'noise_slots': ['intro', 'conjunction', 'intensifier', 'conjunction', 'noise'],
        'min_symptoms': 3
    },
    {
        'template': '{noise} {s1}, {s2} {conjunction} {s3} {intensifier}. {time}',
        'noise_slots': ['noise', 'conjunction', 'intensifier', 'time'],
        'min_symptoms': 3
    },
    
    # ===== ОТРИЦАНИЯ С ШУМОМ =====
    {
        'template': '{intro} {negation_phrase}, {conjunction} {s2} {intensifier}',
        'noise_slots': ['intro', 'negation_phrase', 'conjunction', 'intensifier'],
        'min_symptoms': 2,
        'negation_template': True
    },
    {
        'template': '{s1} {intensifier}, {conjunction} {intro} {negation_phrase}',
        'noise_slots': ['intensifier', 'conjunction', 'intro', 'negation_phrase'],
        'min_symptoms': 2,
        'negation_template': True
    },
    {
        'template': '{negation_phrase}, {conjunction} {s2} {intensifier} {time}',
        'noise_slots': ['negation_phrase', 'conjunction', 'intensifier', 'time'],
        'min_symptoms': 2,
        'negation_template': True
    },
]

# ====================== ФУНКЦИЯ ЗАГРУЗКИ ИЗ БД ======================

def load_symptoms_from_db():
    """Загружает список симптомов из PostgreSQL."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT id, name FROM public.diagnosis_symptom ORDER BY id")
        symptoms = [{'id': row['id'], 'name': row['name'].strip()} for row in cur.fetchall()]
        cur.close()
        conn.close()
        print(f"Загружено {len(symptoms)} симптомов из БД")
        return symptoms
    except Exception as e:
        print(f"Ошибка подключения к БД: {e}")
        return None

# ====================== ЗАГРУЗКА СЛОВАРЯ ======================

def load_synonyms(file_path: str) -> Dict[str, List[str]]:
    """Загружает словарь синонимов из JSON-файла."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            synonyms = json.load(f)
        print(f"Загружен словарь синонимов: {len(synonyms)} симптомов")
        return synonyms
    except Exception as e:
        print(f"Ошибка загрузки словаря синонимов: {e}")
        return None

# ====================== ФУНКЦИИ АУГМЕНТАЦИИ ======================

def introduce_typo(word: str) -> str:
    """Вносит случайную опечатку в слово."""
    if len(word) < 3:
        return word
    
    typo_type = random.choice(['delete', 'replace', 'transpose', 'insert'])
    
    if typo_type == 'delete' and len(word) > 2:
        pos = random.randint(0, len(word)-1)
        return word[:pos] + word[pos+1:]
    
    elif typo_type == 'replace':
        pos = random.randint(0, len(word)-1)
        replacements = {
            'а': 'о', 'о': 'а', 'у': 'е', 'е': 'у', 
            'к': 'л', 'л': 'к', 'н': 'т', 'т': 'н',
            'р': 'п', 'п': 'р', 'с': 'д', 'д': 'с',
            'и': 'й', 'й': 'и', 'б': 'ю', 'ю': 'б'
        }
        if word[pos] in replacements:
            return word[:pos] + replacements[word[pos]] + word[pos+1:]
        return word
    
    elif typo_type == 'transpose' and len(word) > 3:
        pos = random.randint(0, len(word)-2)
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    
    elif typo_type == 'insert':
        pos = random.randint(0, len(word))
        char_to_insert = random.choice('абвгдежзиклмнопрстуфхцчшщъыьэюя')
        return word[:pos] + char_to_insert + word[pos:]
    
    return word

def keyboard_layout_error(word: str) -> str:
    """Имитирует ошибку раскладки (ru -> en)."""
    layout_map = {
        'а': 'f', 'б': ',', 'в': 'd', 'г': 'u', 'д': 'l', 'е': 't',
        'ё': '`', 'ж': ';', 'з': 'p', 'и': 'b', 'й': 'q', 'к': 'r',
        'л': 'k', 'м': 'v', 'н': 'y', 'о': 'j', 'п': 'g', 'р': 'h',
        'с': 'c', 'т': 'n', 'у': 'e', 'ф': 'a', 'х': '[', 'ц': 'w',
        'ч': 'x', 'ш': 'i', 'щ': 'o', 'ъ': ']', 'ы': 's', 'ь': 'm',
        'э': "'", 'ю': '.', 'я': 'z', ' ': ' '
    }
    result = []
    for ch in word:
        if ch.lower() in layout_map:
            mapped = layout_map[ch.lower()]
            result.append(mapped.upper() if ch.isupper() else mapped)
        else:
            result.append(ch)
    return ''.join(result)

def get_noise_by_length(length_type: str) -> str:
    """Возвращает шум нужной длины."""
    if length_type == 'short':
        min_len, max_len = SHORT_NOISE_LEN
    elif length_type == 'medium':
        min_len, max_len = MEDIUM_NOISE_LEN
    else:  # long
        min_len, max_len = LONG_NOISE_LEN
    
    target_len = random.randint(min_len, max_len)
    
    # Смешиваем разные типы шума для достижения нужной длины
    noise_parts = []
    current_len = 0
    
    while current_len < target_len:
        if random.random() < 0.3 and len(noise_parts) > 0:
            # Добавляем союз
            part = random.choice(CONJUNCTIONS)
        elif random.random() < 0.4:
            # Добавляем фразу из NOISE_PHRASES
            part = random.choice(NOISE_PHRASES)
        else:
            # Добавляем отдельные слова
            part = ' '.join(random.choices(EXTRA_NOISE_WORDS, k=random.randint(1, 3)))
        
        noise_parts.append(part)
        current_len += len(part.split())
    
    return ' '.join(noise_parts)

# ====================== ОСНОВНОЙ ГЕНЕРАТОР ======================

class SymptomPhraseGenerator:
    def __init__(self, symptoms: List[Dict], synonyms: Dict[str, List[str]]):
        self.symptoms = symptoms
        self.synonyms = synonyms
        self.name_to_id = {s['name']: s['id'] for s in symptoms}
        self.symptom_usage = {s['id']: 0 for s in symptoms}
        
        # Загружаем корреляции симптомов
        self.cooccurrence, self.symptom_pairs = load_disease_symptom_correlations()
        self.symptom_id_to_index = {s['id']: i for i, s in enumerate(symptoms)}
        
    def get_synonyms_for_symptom(self, symptom_name: str) -> List[str]:
        """Возвращает синонимы для симптома."""
        if symptom_name in self.synonyms:
            return self.synonyms[symptom_name]
        else:
            return [symptom_name]
    
    def get_correlated_symptoms(self, base_symptom_id: int, n: int) -> List[int]:
        """
        Возвращает симптомы, коррелирующие с base_symptom_id.
        """
        if not self.cooccurrence or n <= 1:
            return []
        
        # Находим все пары с base_symptom_id
        correlated = []
        for (s1, s2), count in self.cooccurrence.items():
            if s1 == base_symptom_id:
                correlated.append((s2, count))
            elif s2 == base_symptom_id:
                correlated.append((s1, count))
        
        # Сортируем по частоте совместной встречаемости
        correlated.sort(key=lambda x: x[1], reverse=True)
        
        # Берем топ-n симптомов
        return [s_id for s_id, _ in correlated[:n]]
    
    def get_random_symptoms(self, n: int, negation_indices: List[int]) -> List[Dict]:
        """Выбирает симптомы с учетом корреляций."""
        if n == 0:
            return []
        
        selected = []
        
        # Выбираем первый симптом случайно
        first = random.choice(self.symptoms)
        selected.append(first['id'])
        
        # Для остальных пытаемся найти коррелирующие
        for _ in range(n - 1):
            # С вероятностью 70% берем коррелирующий симптом
            if random.random() < 0.7 and self.cooccurrence:
                correlated = self.get_correlated_symptoms(selected[-1], 5)
                # Фильтруем уже выбранные
                correlated = [s for s in correlated if s not in selected]
                if correlated:
                    selected.append(random.choice(correlated))
                    continue
            
            # Иначе берем случайный
            remaining = [s for s in self.symptoms if s['id'] not in selected]
            if remaining:
                selected.append(random.choice(remaining)['id'])
            else:
                # Если закончились, добираем случайными
                selected.append(random.choice(self.symptoms)['id'])
        
        # Преобразуем ID в объекты симптомов
        result = []
        for i, symptom_id in enumerate(selected):
            symptom = next(s for s in self.symptoms if s['id'] == symptom_id)
            result.append({
                'id': symptom['id'],
                'name': symptom['name'],
                'negated': i in negation_indices,
                'used_text': None
            })
            self.symptom_usage[symptom['id']] += 1
        
        return result
    
    def apply_synonyms(self, symptoms: List[Dict]) -> List[Dict]:
        """Заменяет канонические названия на синонимы."""
        result = []
        for symptom in symptoms:
            symptom_copy = symptom.copy()
            synonym_list = self.get_synonyms_for_symptom(symptom['name'])
            
            if len(synonym_list) > 1 and random.random() < SYNONYM_PROBABILITY:
                synonym = random.choice([s for s in synonym_list if s != symptom['name']])
                symptom_copy['used_text'] = synonym
            else:
                symptom_copy['used_text'] = symptom['name']
            
            result.append(symptom_copy)
        return result
    
    def format_phrase(self, template: Dict, symptoms: List[Dict]) -> str:
        """Форматирует фразу по шаблону."""
        format_dict = {}
        
        # Подставляем симптомы
        for i, symptom in enumerate(symptoms):
            format_dict[f's{i+1}'] = symptom['used_text']
        
        # Если симптомов меньше, чем ожидает шаблон, добавляем заглушки
        matches = re.findall(r's(\d+)', template['template'])
        if matches:
            max_symptom_num = max(int(num) for num in matches)
            for i in range(len(symptoms) + 1, max_symptom_num + 1):
                format_dict[f's{i}'] = ''
        
        # Определяем длину шума
        noise_length = random.choices(
            list(NOISE_LENGTH_PROBS.keys()),
            weights=list(NOISE_LENGTH_PROBS.values())
        )[0]
        
        # Заполняем слоты шума
        for slot in template['noise_slots']:
            if slot == 'intro':
                format_dict['intro'] = random.choice(INTRO_WORDS)
            elif slot == 'intensifier':
                format_dict['intensifier'] = random.choice(INTENSIFIERS)
            elif slot == 'conjunction':
                format_dict['conjunction'] = random.choice(CONJUNCTIONS)
            elif slot == 'time':
                format_dict['time'] = random.choice(TIME_MARKERS)
            elif slot == 'noise':
                format_dict['noise'] = get_noise_by_length(noise_length)
            elif slot == 'extra':
                format_dict['extra'] = random.choice(EXTRA_NOISE_WORDS)
            elif slot == 'negation_phrase':
                # Выбираем случайную фразу с отрицанием
                neg_phrase = random.choice(NEGATION_PHRASES)
                # Находим первый отрицаемый симптом
                negated_symptoms = [s for s in symptoms if s['negated']]
                if negated_symptoms:
                    # Подставляем в неё первый отрицаемый симптом
                    neg_phrase = neg_phrase.replace('{s}', negated_symptoms[0]['used_text'])
                else:
                    # Если нет отрицаемых симптомов, вставляем шум
                    neg_phrase = neg_phrase.replace('{s}', random.choice(EXTRA_NOISE_WORDS))
                format_dict['negation_phrase'] = neg_phrase
        
        # Проверяем, все ли слоты заполнены
        all_slots = re.findall(r'{(\w+)}', template['template'])
        for slot in all_slots:
            if slot not in format_dict:
                format_dict[slot] = ''
        
        # Формируем фразу
        try:
            phrase = template['template'].format(**format_dict)
            # Убираем лишние пробелы
            phrase = re.sub(r'\s+', ' ', phrase)
            phrase = phrase.strip()
            # Убираем двойные пробелы после знаков препинания
            phrase = re.sub(r'\s+([,.!?])', r'\1', phrase)
            # Делаем первую букву заглавной
            if phrase:
                phrase = phrase[0].upper() + phrase[1:]
        except KeyError as e:
            print(f"Ошибка форматирования: {e}")
            phrase = template['template']
        
        return phrase
    
    def apply_augmentations(self, phrase: str, symptoms: List[Dict], 
                           template: Dict) -> List[Dict]:
        """Применяет аугментации."""
        variants = [{
            'text': phrase,
            'augmentations': {},
            'symptoms': symptoms,
            'template': template['template']
        }]
        
        # Опечатки
        if random.random() < TYPO_RATIO:
            words = phrase.split()
            typo_words = []
            for word in words:
                if random.random() < 0.2 and len(word) > 3:
                    typo_words.append(introduce_typo(word))
                else:
                    typo_words.append(word)
            variants.append({
                'text': ' '.join(typo_words),
                'augmentations': {'has_typo': True},
                'symptoms': symptoms,
                'template': template['template']
            })
        
        # Ошибка раскладки
        if random.random() < KEYBOARD_ERROR_RATIO:
            words = phrase.split()
            kb_words = []
            for word in words:
                if random.random() < 0.3:
                    kb_words.append(keyboard_layout_error(word))
                else:
                    kb_words.append(word)
            variants.append({
                'text': ' '.join(kb_words),
                'augmentations': {'keyboard_error': True},
                'symptoms': symptoms,
                'template': template['template']
            })
        
        return variants
    
    def generate(self, n_phrases: int) -> List[Dict]:
        """Генерирует фразы."""
        random.seed(RANDOM_SEED)
        phrase_id = 1
        negation_count = 0
        max_negations = int(n_phrases * NEGATION_RATIO)
        
        generated_phrases = []
        
        print(f"\nЦель: {n_phrases} фраз")
        print(f"Максимум отрицаний: {max_negations}")
        
        while len(generated_phrases) < n_phrases:
            # Определяем количество симптомов
            n_symptoms = random.choices(
                list(SYMPTOM_COUNT_PROBS.keys()),
                weights=list(SYMPTOM_COUNT_PROBS.values())
            )[0]
            
            # Определяем, будут ли отрицания
            has_negation = (random.random() < NEGATION_RATIO and 
                          negation_count < max_negations and
                          n_symptoms > 0)
            
            # Выбираем подходящий шаблон
            if has_negation:
                # Шаблоны с отрицанием
                negation_templates = [t for t in TEMPLATES if t.get('negation_template')]
                if negation_templates:
                    template = random.choice(negation_templates)
                    # Отрицаем первый симптом
                    negation_indices = [0]
                else:
                    template = random.choice(TEMPLATES)
                    negation_indices = []
            else:
                # Обычные шаблоны
                suitable_templates = [t for t in TEMPLATES 
                                    if not t.get('negation_template') 
                                    and t['min_symptoms'] <= n_symptoms]
                template = random.choice(suitable_templates) if suitable_templates else random.choice(TEMPLATES)
                negation_indices = []
            
            # Генерируем симптомы
            symptoms = self.get_random_symptoms(n_symptoms, negation_indices)
            symptoms = self.apply_synonyms(symptoms)
            
            # Формируем фразу
            phrase = self.format_phrase(template, symptoms)
            
            # Применяем аугментации
            variants = self.apply_augmentations(phrase, symptoms, template)
            
            for variant in variants:
                if len(generated_phrases) >= n_phrases:
                    break
                
                generated_phrases.append({
                    'phrase_id': phrase_id,
                    'phrase_text': variant['text'],
                    'template': variant['template'],
                    'symptoms': [
                        {
                            'canonical_id': s['id'],
                            'canonical_name': s['name'],
                            'present': not s['negated'],
                            'negated': s['negated'],
                            'used_text': s['used_text']
                        }
                        for s in variant['symptoms']
                    ],
                    'augmentations': variant['augmentations']
                })
                phrase_id += 1
            
            if len(generated_phrases) % 200 == 0:
                print(f"   Сгенерировано {len(generated_phrases)} фраз...")
        
        return generated_phrases

# ====================== ЗАПУСК ======================

def main():
    print("=" * 60)
    print("ГЕНЕРАТОР ДАТАСЕТА ДЛЯ ИЗВЛЕЧЕНИЯ СИМПТОМОВ v5.0")
    print("=" * 60)
    
    symptoms = load_symptoms_from_db()
    if not symptoms:
        return
    
    synonyms = load_synonyms(SYNONYMS_PATH)
    if not synonyms:
        return
    
    generator = SymptomPhraseGenerator(symptoms, synonyms)
    
    print(f"\n🚀 Генерация {N_PHRASES_TOTAL} фраз...")
    phrases = generator.generate(N_PHRASES_TOTAL)
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(phrases, f, ensure_ascii=False, indent=2)
    
    print(f"\nГотово! Сохранено в: {OUTPUT_PATH}")
    
    # Статистика по токенам
    total_tokens = 0
    symptom_tokens = 0
    
    for p in phrases:
        tokens = p['phrase_text'].split()
        total_tokens += len(tokens)
        for s in p['symptoms']:
            symptom_tokens += len(s['used_text'].split())
    
    print("\n" + "=" * 60)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 60)
    print(f"Всего фраз: {len(phrases)}")
    print(f"Всего токенов: {total_tokens}")
    print(f"Токенов-симптомов: {symptom_tokens}")
    print(f"Доля симптомов: {symptom_tokens/total_tokens*100:.1f}%")
    
    # Считаем отрицания
    negations = sum(1 for p in phrases if any(s['negated'] for s in p['symptoms']))
    print(f"\nОТРИЦАНИЯ:")
    print(f"Фраз с отрицаниями: {negations} ({negations/len(phrases)*100:.1f}%)")
    
    # Считаем аугментации
    typos = sum(1 for p in phrases if p['augmentations'].get('has_typo'))
    kb_errors = sum(1 for p in phrases if p['augmentations'].get('keyboard_error'))
    
    print(f"\nАУГМЕНТАЦИИ:")
    print(f"С опечатками: {typos} ({typos/len(phrases)*100:.1f}%)")
    print(f"С ошибками раскладки: {kb_errors} ({kb_errors/len(phrases)*100:.1f}%)")
    
    # Показываем примеры
    print("\nПРИМЕРЫ СГЕНЕРИРОВАННЫХ ФРАЗ:")
    for i, p in enumerate(phrases[:5]):
        symptoms_list = ', '.join([f"{s['used_text']}({'-' if s['negated'] else '+'})" 
                                  for s in p['symptoms']])
        print(f"\n{i+1}. {p['phrase_text']}")
        print(f"   Симптомы: {symptoms_list}")
        if p['augmentations']:
            print(f"   Аугментации: {p['augmentations']}")

if __name__ == "__main__":
    main()