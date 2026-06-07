# check_seqeval_single_class.py
import sys
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

def load_conll_labels(filepath):
    """
    Загружает CONLL-файл и возвращает список предложений,
    каждое предложение — список меток (строк).
    Поддерживает разделение по табуляции или пробелу.
    """
    sentences = []
    current = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            # Пытаемся разделить по табуляции, если нет — по пробелу
            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split()
            # Ожидаем минимум два поля: токен и метка
            if len(parts) >= 2:
                label = parts[-1]   # последнее поле — метка
                current.append(label)
        if current:
            sentences.append(current)
    return sentences

def main():
    if len(sys.argv) != 3:
        print("Usage: python check_seqeval_single_class.py <gold_file> <pred_file>")
        print("Example: python check_seqeval_single_class.py test_10.conll predicted_10.conll")
        sys.exit(1)

    gold_path = sys.argv[1]
    pred_path = sys.argv[2]

    gold = load_conll_labels(gold_path)
    pred = load_conll_labels(pred_path)

    if len(gold) != len(pred):
        print(f"Предупреждение: количество предложений разное (gold: {len(gold)}, pred: {len(pred)})")
        min_len = min(len(gold), len(pred))
        gold = gold[:min_len]
        pred = pred[:min_len]

    # Проверка совпадения длин каждого предложения
    mismatch = False
    for i, (g, p) in enumerate(zip(gold, pred)):
        if len(g) != len(p):
            print(f"Ошибка: предложение {i+1} имеет разную длину (gold: {len(g)}, pred: {len(p)})")
            mismatch = True
    if mismatch:
        print("Невозможно вычислить метрики из-за несовпадения длин. Проверьте файлы.")
        sys.exit(1)

    report = classification_report(gold, pred, scheme=IOB2, output_dict=True)

    print("\n=== РЕЗУЛЬТАТЫ seqeval ===")
    print(f"Micro F1:    {report['micro avg']['f1-score']:.4f}")
    print(f"Macro F1:    {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    print(f"\nРазница micro-macro: {abs(report['micro avg']['f1-score'] - report['macro avg']['f1-score']):.10f}")

if __name__ == '__main__':
    main()