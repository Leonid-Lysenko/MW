# ml/nlp/extractors/rule_based.py
import json
import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from .base import SymptomExtractor

# Определяем режим работы по переменной окружения
DEBUG = os.environ.get("DEBUG", "True") == "True"

# Импортируем connection только если нужна БД
if DEBUG:
    from django.db import connection


class RuleBasedExtractor(SymptomExtractor):
    """
    Rule-based экстрактор симптомов.
    
    Использует словарь синонимов для поиска симптомов по точному совпадению
    и лингвистические правила для обработки отрицаний.
    
    Поддерживает множественное соответствие: один синоним может относиться
    к нескольким каноническим симптомам.
    
    В режиме DEBUG=True загружает симптомы из БД.
    В режиме DEBUG=False (продакшен) загружает симптомы из JSON.
    """
    
    def __init__(self, synonyms_path: Optional[str] = None):
        """Инициализирует экстрактор, загружая словарь симптомов."""
        super().__init__()
        
        # Новая структура: {фраза: [(canonical_name1, symptom_id1), ...]}
        self.symptom_dict: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.canonical_to_id: Dict[str, int] = {}
        
        # Путь к файлу синонимов
        if synonyms_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synonyms_path = os.path.join(base_dir, 'data', 'synonyms', 'synonyms_updated.json')
        
        if DEBUG:
            # Локальная разработка: загружаем из БД
            print("[RuleBasedExtractor] DEBUG mode: loading from database")
            self._load_symptoms_from_db()
            self._load_synonyms(synonyms_path)
        else:
            # Продакшен (Render): загружаем из JSON
            print("[RuleBasedExtractor] PRODUCTION mode: loading from JSON")
            self._load_synonyms_from_json(synonyms_path)
        
        # Сортируем ключи по убыванию длины для поиска самых длинных совпадений
        self.sorted_phrases = sorted(self.symptom_dict.keys(), key=len, reverse=True)
        
        # Список слов отрицания
        self.negation_words = [
            'нет', 'не', 'без', 'отсутствует', 'кроме', 'ни',
            'прошел', 'прошла', 'прошло', 'прошли',
            'перестал', 'перестала', 'перестало', 'перестали',
            'исчез', 'исчезла', 'исчезло', 'исчезли',
            'пропал', 'пропала', 'пропало', 'пропали',
            'прекратился', 'прекратилась', 'прекратилось', 'прекратились'
        ]
        
        # Конструкции, которые НЕ являются отрицанием
        self.non_negation_patterns = [
            (r'не\s+только\s+', 'positive'),
            (r'не\s+могу\s+', 'positive'),
            (r'не\s+знаю\s+', 'positive'),
        ]
        
        # Союзы, которые разделяют части предложения
        self.clause_separators = ['но', 'а', 'однако', 'зато']
    
    def _load_symptoms_from_db(self):
        """Загружает канонические симптомы из БД (только для локальной разработки)."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, name FROM diagnosis_symptom")
                rows = cursor.fetchall()
                for symptom_id, name in rows:
                    name_lower = name.lower()
                    self.symptom_dict[name_lower].append((name, symptom_id))
                    self.canonical_to_id[name] = symptom_id
        except Exception as e:
            print(f"Ошибка загрузки симптомов из БД: {e}")
    
    def _load_synonyms(self, synonyms_path: str):
        """Загружает словарь синонимов из JSON и связывает с ID из БД (для локальной разработки)."""
        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            for canonical_name, synonyms in synonyms_data.items():
                symptom_id = self.canonical_to_id.get(canonical_name)
                if symptom_id is None:
                    for name, sid in self.canonical_to_id.items():
                        if name.lower() == canonical_name.lower():
                            symptom_id = sid
                            canonical_name = name
                            break
                    if symptom_id is None:
                        print(f" Предупреждение: симптом '{canonical_name}' не найден в БД")
                        continue
                
                for synonym in synonyms:
                    if not synonym or not synonym.strip():
                        continue
                    
                    synonym_clean = synonym.strip().lower()
                    existing_pairs = self.symptom_dict[synonym_clean]
                    if (canonical_name, symptom_id) not in existing_pairs:
                        existing_pairs.append((canonical_name, symptom_id))
                    
        except FileNotFoundError:
            print(f"Предупреждение: файл {synonyms_path} не найден")
        except Exception as e:
            print(f"Ошибка загрузки синонимов: {e}")
    
    def _load_synonyms_from_json(self, synonyms_path: str):
        """Загружает словарь синонимов из JSON и создаёт ID симптомов (для Render)."""
        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            symptom_id = 1
            for canonical_name, synonyms in synonyms_data.items():
                self.canonical_to_id[canonical_name] = symptom_id
                # Добавляем сам симптом как синоним
                self.symptom_dict[canonical_name.lower()].append((canonical_name, symptom_id))
                for synonym in synonyms:
                    if synonym and synonym.strip():
                        synonym_clean = synonym.strip().lower()
                        existing = self.symptom_dict[synonym_clean]
                        if (canonical_name, symptom_id) not in existing:
                            existing.append((canonical_name, symptom_id))
                symptom_id += 1
                    
        except Exception as e:
            print(f"Ошибка загрузки синонимов: {e}")
    
    def _find_negation_positions(self, text: str) -> List[Dict]:
        """Находит все позиции слов отрицания в тексте."""
        positions = []
        text_lower = text.lower()
        
        for word in self.negation_words:
            start = 0
            while True:
                pos = text_lower.find(word, start)
                if pos == -1:
                    break
                
                before = pos == 0 or not text_lower[pos-1].isalnum()
                after = (pos + len(word) == len(text_lower) or 
                        not text_lower[pos + len(word)].isalnum())
                
                if before and after:
                    positions.append({
                        'start': pos,
                        'end': pos + len(word),
                        'word': word
                    })
                start = pos + 1
        
        return positions
    
    def _is_non_negation_construction(self, text: str, symptom_pos: int) -> bool:
        """Проверяет, является ли найденное "не" частью конструкции, которая не означает отрицание симптома."""
        text_lower = text.lower()
        
        not_only_pattern = r'не\s+только\s+'
        match = re.search(not_only_pattern, text_lower[max(0, symptom_pos-30):symptom_pos])
        if match:
            return True
        
        if 'не могу' in text_lower[max(0, symptom_pos-20):symptom_pos]:
            return True
        
        return False
    
    def _find_clause_boundaries(self, text: str, symptom_pos: int) -> tuple:
        """Находит границы предложения или придаточной части, в которой находится симптом."""
        text_lower = text.lower()
        
        start = 0
        for separator in self.clause_separators:
            sep_pos = text_lower.rfind(' ' + separator + ' ', 0, symptom_pos)
            if sep_pos != -1:
                start = sep_pos + len(separator) + 2
                break
        
        end = len(text)
        for separator in self.clause_separators:
            sep_pos = text_lower.find(' ' + separator + ' ', symptom_pos)
            if sep_pos != -1 and sep_pos < end:
                end = sep_pos
        
        return start, end
    
    def _check_negation(self, symptom_start: int, symptom_end: int, 
                        negation_positions: List[Dict], text: str) -> bool:
        """
        Проверяет, относится ли отрицание к симптому.
        Учитывает запятые как разделители перечислений.
        """
        clause_start = symptom_start
        for i in range(symptom_start - 1, -1, -1):
            char = text[i]
            if char in [',', ';', '.', '!', '?']:
                break
            if i >= 2:
                prev_chars = text[i-2:i+1].lower()
                if prev_chars in [' и ', ' а ', ' но ', ' или ']:
                    break
            clause_start = i
        
        clause_end = symptom_end
        for i in range(symptom_end, len(text)):
            char = text[i]
            if char in [',', ';', '.', '!', '?']:
                break
            clause_end = i + 1
        
        for neg in negation_positions:
            neg_start = neg['start']
            neg_word = neg['word']
            
            if clause_start <= neg_start <= clause_end:
                comma_between = False
                for pos in range(neg_start, symptom_start):
                    if pos < len(text) and text[pos] == ',':
                        comma_between = True
                        break
                
                if not comma_between:
                    return True
        
        return False
    
    def check_symptom_negation(self, text: str, symptom_name: str) -> str:
        """
        Публичный метод для проверки отрицания конкретного симптома.
        Учитывает запятые и перечисления.
        """
        import re
        
        text_lower = text.lower()
        symptom_lower = symptom_name.lower()
        
        text_normalized = re.sub(r'\s+(и|а|но|или)\s+', ';', text_lower)
        text_normalized = re.sub(r'\s*,\s*', ';', text_normalized)
        
        clauses = [c.strip() for c in text_normalized.split(';') if c.strip()]
        
        for clause in clauses:
            if symptom_lower in clause:
                for neg_word in self.negation_words:
                    if neg_word in clause:
                        neg_pos = clause.find(neg_word)
                        symptom_pos = clause.find(symptom_lower)
                        if abs(neg_pos - symptom_pos) < 30:
                            return 'absent'
                return 'present'
        
        return 'present'
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает симптомы из текста.
        Возвращает ВСЕ уникальные симптомы, соответствующие найденным синонимам.
        """
        if not text:
            return []
        
        search_text = text.lower()
        negation_positions = self._find_negation_positions(text)
        
        position_groups = {}
        
        for phrase in self.sorted_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in search_text:
                start = 0
                while True:
                    pos = search_text.find(phrase_lower, start)
                    if pos == -1:
                        break
                    
                    key = (pos, pos + len(phrase_lower))
                    if key not in position_groups:
                        position_groups[key] = []
                    
                    for canonical_name, symptom_id in self.symptom_dict.get(phrase, []):
                        existing_ids = [m['symptom_id'] for m in position_groups[key]]
                        if symptom_id not in existing_ids:
                            position_groups[key].append({
                                'start': pos,
                                'end': pos + len(phrase_lower),
                                'canonical_name': canonical_name,
                                'symptom_id': symptom_id,
                                'matched_text': phrase,
                                'original_phrase': phrase
                            })
                    start = pos + 1
        
        sorted_keys = sorted(position_groups.keys(), 
                            key=lambda x: (x[1] - x[0]), 
                            reverse=True)
        
        used_positions = set()
        final_matches = []
        
        for start, end in sorted_keys:
            overlap = False
            for pos in range(start, end):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                for pos in range(start, end):
                    used_positions.add(pos)
                final_matches.extend(position_groups[(start, end)])
        
        for match in final_matches:
            is_negated = self._check_negation(
                match['start'], 
                match['end'], 
                negation_positions,
                text
            )
            match['status'] = 'absent' if is_negated else 'present'
            match['confidence'] = 1.0
        
        unique_by_symptom = {}
        for match in final_matches:
            sid = match['symptom_id']
            if sid not in unique_by_symptom:
                unique_by_symptom[sid] = match
            else:
                existing = unique_by_symptom[sid]
                if match['status'] == 'present' and existing['status'] == 'absent':
                    unique_by_symptom[sid] = match
        
        result = []
        for match in unique_by_symptom.values():
            result.append({
                'symptom_id': match['symptom_id'],
                'canonical_name': match['canonical_name'],
                'status': match['status'],
                'confidence': match['confidence'],
                'matched_text': match['original_phrase']
            })
        
        return result
    
    def _resolve_overlaps(self, matches: List[Dict]) -> List[Dict]:
        """
        Разрешает перекрытия между найденными симптомами.
        """
        if not matches:
            return []
        
        matches.sort(key=lambda x: (x['end'] - x['start']), reverse=True)
        
        used_positions = set()
        filtered = []
        
        for match in matches:
            overlap = False
            for pos in range(match['start'], match['end']):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                for pos in range(match['start'], match['end']):
                    used_positions.add(pos)
                filtered.append(match)
        
        filtered.sort(key=lambda x: x['start'])
        return filtered
    
    def get_synonym_mappings(self, phrase: str) -> List[Tuple[str, int]]:
        """Возвращает все канонические симптомы для данного синонима."""
        return self.symptom_dict.get(phrase.lower(), [])
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Возвращает статистику покрытия словаря."""
        return {
            'total_phrases': len(self.symptom_dict),
            'total_mappings': sum(len(v) for v in self.symptom_dict.values()),
            'unique_symptoms': len(self.canonical_to_id),
        }