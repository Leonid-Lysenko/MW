# ml/nlp/extractors/rule_based.py
import json
import os
from typing import List, Dict, Any, Optional
from django.db import connection
from .base import SymptomExtractor
from ..utils.text_processing import normalize_text, find_negations

class RuleBasedExtractor(SymptomExtractor):
    """
    Rule-based экстрактор симптомов.
    
    Использует словарь синонимов для поиска симптомов по точному совпадению
    и лингвистические правила для обработки отрицаний.
    
    Ограничения:
    - Не обрабатывает падежи (требуется точное совпадение)
    - Не обрабатывает опечатки
    - Зависит от полноты словаря синонимов
    """
    
    def __init__(self, synonyms_path: Optional[str] = None):
        """
        Инициализирует экстрактор, загружая словарь симптомов.
        
        Args:
            synonyms_path: путь к файлу synonyms.json.
                          Если None, используется путь по умолчанию.
        """
        super().__init__()
        self.symptom_dict = {}  # {фраза: (canonical_name, symptom_id)}
        self.canonical_to_id = {}  # {canonical_name: symptom_id}
        
        # Загружаем симптомы из БД
        self._load_symptoms_from_db()
        
        # Загружаем синонимы
        if synonyms_path is None:
            # Путь относительно текущего файла
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            synonyms_path = os.path.join(base_dir, 'data', 'synonyms', 'synonyms.json')
        
        self._load_synonyms(synonyms_path)
        
        # Сортируем ключи по убыванию длины для поиска самых длинных совпадений
        self.sorted_phrases = sorted(self.symptom_dict.keys(), key=len, reverse=True)
        
        print(f"[RuleBasedExtractor] Загружено {len(self.symptom_dict)} фраз-синонимов")
        print(f"[RuleBasedExtractor] Покрыто симптомов: {len(self.canonical_to_id)}")
    
    def _load_symptoms_from_db(self):
        """Загружает канонические симптомы из БД."""
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT id, name FROM diagnosis_symptom")
                rows = cursor.fetchall()
                for symptom_id, name in rows:
                    # Каноническое имя тоже добавляем в словарь
                    # Важно: не нормализуем, оставляем как есть для точного совпадения
                    self.symptom_dict[name] = (name, symptom_id)
                    self.canonical_to_id[name] = symptom_id
        except Exception as e:
            print(f"Ошибка загрузки симптомов из БД: {e}")
    
    def _load_synonyms(self, synonyms_path: str):
        """Загружает словарь синонимов из JSON."""
        try:
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            for canonical_name, synonyms in synonyms_data.items():
                symptom_id = self.canonical_to_id.get(canonical_name)
                if symptom_id is None:
                    # Если симптом не найден в БД — пропускаем
                    continue
                
                # Добавляем каждый синоним
                for synonym in synonyms:
                    if synonym:  # не пустой
                        # Сохраняем синоним как есть, без нормализации
                        self.symptom_dict[synonym] = (canonical_name, symptom_id)
        except FileNotFoundError:
            print(f"Предупреждение: файл {synonyms_path} не найден")
        except Exception as e:
            print(f"Ошибка загрузки синонимов: {e}")
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Извлекает симптомы из текста.
        """
        if not text:
            return []
        
        # Приводим текст к нижнему регистру для поиска
        search_text = text.lower()
        
        # Поиск всех вхождений
        matches = []
        for phrase in self.sorted_phrases:
            phrase_lower = phrase.lower()
            if phrase_lower in search_text:
                # Находим все вхождения
                start = 0
                while True:
                    pos = search_text.find(phrase_lower, start)
                    if pos == -1:
                        break
                    
                    canonical_name, symptom_id = self.symptom_dict[phrase]
                    matches.append({
                        'start': pos,
                        'end': pos + len(phrase_lower),
                        'canonical_name': canonical_name,
                        'symptom_id': symptom_id,
                        'matched_text': phrase,
                        'original_phrase': phrase
                    })
                    start = pos + 1
        
        # Разрешаем перекрытия (оставляем самые длинные)
        matches = self._resolve_overlaps(matches)
        
        # Обрабатываем отрицания
        for match in matches:
            is_negated = find_negations(
                text.lower(),
                match['start'], 
                match['end']
            )
            match['status'] = 'absent' if is_negated else 'present'
            match['confidence'] = 1.0
        
        # Преобразуем в нужный формат
        result = []
        for match in matches:
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
        Оставляет только самые длинные неперекрывающиеся вхождения.
        """
        if not matches:
            return []
        
        # Сортируем по длине (убывание)
        matches.sort(key=lambda x: (x['end'] - x['start']), reverse=True)
        
        used_positions = set()
        filtered = []
        
        for match in matches:
            # Проверяем, не перекрывается ли с уже добавленными
            overlap = False
            for pos in range(match['start'], match['end']):
                if pos in used_positions:
                    overlap = True
                    break
            
            if not overlap:
                # Добавляем и отмечаем занятые позиции
                for pos in range(match['start'], match['end']):
                    used_positions.add(pos)
                filtered.append(match)
        
        # Возвращаем в порядке появления в тексте
        filtered.sort(key=lambda x: x['start'])
        return filtered
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Возвращает статистику покрытия словаря."""
        return {
            'total_phrases': len(self.symptom_dict),
            'unique_symptoms': len(self.canonical_to_id),
        }