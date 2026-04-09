# C:\Users\Leo\PyWork\MedWebsite\ml\nlp\evaluation\evaluate_all_approaches.py

"""
Сравнительный анализ методов извлечения симптомов.

Сравнивает:
    - Rule-based
    - Семантический поиск (E5-large)
    - NER (RuBioRoBERTa)
    - Гибридный алгоритм

Вычисляет:
    - Micro F1
    - Macro F1
    - Weighted F1

Вход:  test_set_500.json
Выход: test_results.json + таблица в консоль

Запуск: python evaluate_all_approaches.py
"""

import json
import os
import sys
from typing import List, Dict, Set, Tuple
from collections import defaultdict

# Определяем пути
script_dir = os.path.dirname(os.path.abspath(__file__))  # ml/nlp/evaluation
ml_nlp_dir = os.path.dirname(script_dir)  # ml/nlp
ml_dir = os.path.dirname(ml_nlp_dir)  # ml
project_root = os.path.dirname(ml_dir)  # MedWebsite

# Добавляем пути для импортов
sys.path.insert(0, project_root)  # для импорта ml.*
sys.path.insert(0, os.path.join(project_root, 'backend'))  # для импорта diagnosis, medical_site

print(f"Project root: {project_root}")
print(f"Backend path: {os.path.join(project_root, 'backend')}")

# Настройка Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_site.settings')

try:
    import django
    django.setup()
    print(" Django настроен")
    
    from django.db import connection
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM diagnosis_symptom")
        count = cursor.fetchone()[0]
        print(f" База данных доступна, симптомов: {count}")
except Exception as e:
    print(f" Ошибка настройки Django: {e}")
    print("Продолжаем без Django (экстракторы могут не работать)")

# Импортируем экстракторы
try:
    from ml.nlp.extractors.rule_based import RuleBasedExtractor
    from ml.nlp.extractors.semantic_search import SemanticSearchExtractor
    from ml.nlp.extractors.ner_rubio_finetuned import RuBioRobertaNERExtractor
    from ml.nlp.extractors.hybrid import HybridExtractor
    print("✓ Экстракторы импортированы")
    
    DEEPPAVLOV_AVAILABLE = False
    try:
        from ml.nlp.extractors.ner_deeppavlov import DeepPavlovNERExtractor
        DEEPPAVLOV_AVAILABLE = True
        print(" DeepPavlov доступен")
    except ImportError:
        print(" DeepPavlov не установлен, пропускаем этот подход")
        
except Exception as e:
    print(f" Ошибка импорта: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def load_test_set(path: str) -> List[Dict]:
    """Загружает тестовый набор."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл {path} не найден")
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_metrics(gold: Set[Tuple[int, str]], pred: Set[Tuple[int, str]]) -> Dict[str, float]:
    """
    Считает метрики на уровне (symptom_id, status).
    Возвращает TP, FP, FN для последующего агрегирования.
    """
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    
    # Для отдельных фраз вычисляем метрики (нужны для macro)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def extract_rule_based(extractor, text: str) -> Set[Tuple[int, str]]:
    """Извлекает симптомы rule-based."""
    try:
        results = extractor.extract(text)
        return set((s['symptom_id'], s['status']) for s in results)
    except Exception as e:
        print(f"  Ошибка в rule-based: {e}")
        return set()


def extract_semantic(extractor, text: str) -> Set[Tuple[int, str]]:
    """Извлекает симптомы семантическим поиском."""
    try:
        results = extractor.extract(text)
        return set((s['symptom_id'], 'present') for s in results)
    except Exception as e:
        print(f"  Ошибка в semantic: {e}")
        return set()


def extract_ner(extractor, text: str) -> Set[Tuple[int, str]]:
    """Извлекает симптомы NER."""
    try:
        results = extractor.extract(text)
        return set((s['symptom_id'], 'present') for s in results)
    except Exception as e:
        print(f"  Ошибка в NER: {e}")
        return set()


def extract_hybrid(extractor, text: str) -> Set[Tuple[int, str]]:
    """Извлекает симптомы гибридным подходом."""
    try:
        results = extractor.extract(text)
        return set((s['symptom_id'], s['status']) for s in results)
    except Exception as e:
        print(f"  Ошибка в hybrid: {e}")
        return set()


def evaluate_approach(name: str, extract_func, extractor, test_set: List[Dict]) -> Dict:
    """
    Оценивает один подход на всём тестовом наборе.
    Вычисляет micro, macro и weighted метрики.
    """
    # Для micro: суммируем TP/FP/FN
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Для macro: собираем метрики по каждой фразе
    per_phrase_precision = []
    per_phrase_recall = []
    per_phrase_f1 = []
    
    # Для weighted: учитываем вес
    total_gold_symptoms = 0
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0
    
    errors = []
    n = len(test_set)
    
    for i, item in enumerate(test_set):
        try:
            text = item['phrase_text']
            gold = set((s['symptom_id'], s['status']) for s in item['gold_symptoms'])
            gold_count = len(gold)  # вес для weighted усреднения
            
            pred = extract_func(extractor, text)
            
            metrics = calculate_metrics(gold, pred)
            
            # Micro: суммируем
            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            
            # Macro: добавляем все фразы
            per_phrase_precision.append(metrics['precision'])
            per_phrase_recall.append(metrics['recall'])
            per_phrase_f1.append(metrics['f1'])
            
            # Weighted: накапливаем взвешенные суммы
            total_gold_symptoms += gold_count
            weighted_precision_sum += metrics['precision'] * gold_count
            weighted_recall_sum += metrics['recall'] * gold_count
            weighted_f1_sum += metrics['f1'] * gold_count
                
        except Exception as e:
            errors.append((i, item.get('phrase_id', i), str(e)))
            print(f"  Ошибка на фразе {i}: {e}")
    
    if errors:
        print(f" {len(errors)} ошибок при обработке")
    
    # Micro метрики (из суммарных TP/FP/FN)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Macro метрики (среднее арифметическое по всем фразам)
    macro_precision = sum(per_phrase_precision) / n if n > 0 else 0
    macro_recall = sum(per_phrase_recall) / n if n > 0 else 0
    macro_f1 = sum(per_phrase_f1) / n if n > 0 else 0
    
    # Weighted метрики
    weighted_precision = weighted_precision_sum / total_gold_symptoms if total_gold_symptoms > 0 else 0
    weighted_recall = weighted_recall_sum / total_gold_symptoms if total_gold_symptoms > 0 else 0
    weighted_f1 = weighted_f1_sum / total_gold_symptoms if total_gold_symptoms > 0 else 0
    
    return {
        # Micro
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        # Macro
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        # Weighted
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        # Суммарные
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'errors': len(errors)
    }


def print_detailed_analysis(results: Dict):
    """Выводит детальный анализ результатов."""
    print("ДЕТАЛЬНЫЙ АНАЛИЗ")
    
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"  Micro F1:    {metrics['micro_f1']:.4f}")
        print(f"  Macro F1:    {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Всего TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}")
        if metrics.get('errors', 0) > 0:
            print(f"  Ошибок: {metrics['errors']}")


def main():
    # Определяем пути
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_path = os.path.join(script_dir, 'test_set_500.json')
    results_path = os.path.join(script_dir, 'test_results.json')
    
    print(f"\nСкрипт запущен из: {script_dir}")
    print(f"Тестовый набор: {test_set_path}")
    
    # Загружаем тестовый набор
    print(f"\nЗагрузка тестового набора...")
    try:
        test_set = load_test_set(test_set_path)
        print(f" Загружено {len(test_set)} фраз\n")
    except FileNotFoundError as e:
        print(f" {e}")
        print("Убедитесь, что файл test_set_500.json находится в той же папке, что и скрипт")
        return
    
    # Инициализируем экстракторы
    print("Инициализация экстракторов...")
    
    print("  Загрузка Rule-based...")
    rule_based = RuleBasedExtractor()
    
    print("  Загрузка Semantic Search...")
    semantic = SemanticSearchExtractor(confidence_threshold=0.6)
    
    print("  Загрузка RuBioRoBERTa NER...")
    rubio = RuBioRobertaNERExtractor()
    
    print("  Загрузка Hybrid...")
    hybrid = HybridExtractor()
    
    if DEEPPAVLOV_AVAILABLE:
        print("  Загрузка DeepPavlov NER...")
        deeppavlov = DeepPavlovNERExtractor()
    else:
        deeppavlov = None
    
    print("\n Все экстракторы загружены\n")
    
    # Оцениваем каждый подход
    results = {}
    

    print("НАЧАЛО ТЕСТИРОВАНИЯ")
    
    print("\n[1/4] Оценка Rule-based...")
    results['rule_based'] = evaluate_approach('rule_based', extract_rule_based, rule_based, test_set)
    
    print("\n[2/4] Оценка Semantic Search...")
    results['semantic'] = evaluate_approach('semantic', extract_semantic, semantic, test_set)
    
    print("\n[3/4] Оценка RuBioRoBERTa NER...")
    results['rubio'] = evaluate_approach('rubio', extract_ner, rubio, test_set)
    
    print("\n[4/4] Оценка Hybrid...")
    results['hybrid'] = evaluate_approach('hybrid', extract_hybrid, hybrid, test_set)
    
    if DEEPPAVLOV_AVAILABLE and deeppavlov:
        print("\n[5/5] Оценка DeepPavlov NER...")
        results['deeppavlov'] = evaluate_approach('deeppavlov', extract_ner, deeppavlov, test_set)
    
    # Выводим результаты в таблицу
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print(f"{'Подход':<20} {'Micro F1':<12} {'Macro F1':<12} {'Weighted F1':<12}")
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['micro_f1']:<12.4f} {metrics['macro_f1']:<12.4f} "
              f"{metrics['weighted_f1']:<12.4f}")
    
    # Детальный анализ
    print_detailed_analysis(results)
    
    # Сохраняем результаты
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n Результаты сохранены в {results_path}")


if __name__ == '__main__':
    main()