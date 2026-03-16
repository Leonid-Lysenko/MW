# ml/nlp/scripts/prepare_conll.py
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'labelstudio_ready.json')
OUTPUT_PATH = os.path.join(BASE_DIR, '..', 'data', 'datasets', 'train.conll')

with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

conll_lines = []
for task in tasks:
    text = task['data']['phrase_text']
    annotations = task['annotations'][0]['result']
    
    # Сортируем аннотации по позиции
    annotations.sort(key=lambda x: x['value']['start'])
    
    # Создаем список токенов с метками
    tokens_with_tags = []
    last_end = 0
    
    for ann in annotations:
        start = ann['value']['start']
        end = ann['value']['end']
        label = ann['value']['labels'][0]
        
        # Добавляем текст до сущности (если есть)
        if start > last_end:
            text_before = text[last_end:start].strip()
            if text_before:
                for token in text_before.split():
                    tokens_with_tags.append((token, 'O'))
        
        # Добавляем сущность
        entity_text = text[start:end]
        tokens = entity_text.split()
        for i, token in enumerate(tokens):
            if i == 0:
                tokens_with_tags.append((token, label))
            else:
                # Для продолжения сущности меняем B- на I-
                if label.startswith('B-'):
                    tokens_with_tags.append((token, 'I-' + label[2:]))
                else:
                    tokens_with_tags.append((token, label))
        
        last_end = end
    
    # Добавляем оставшийся текст
    if last_end < len(text):
        text_after = text[last_end:].strip()
        if text_after:
            for token in text_after.split():
                tokens_with_tags.append((token, 'O'))
    
    # Записываем в CONLL формат
    for token, tag in tokens_with_tags:
        conll_lines.append(f"{token}\t{tag}")
    conll_lines.append('')  # пустая строка между предложениями

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(conll_lines))

print(f"CONLL файл сохранён: {OUTPUT_PATH}")