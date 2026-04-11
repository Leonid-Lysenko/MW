#!/bin/bash
# build.sh - скрипт сборки для Render

echo "Начало сборки..."

# Установка зависимостей
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install huggingface-hub

# Предзагрузка E5-small модели (вместо E5-large)
echo "Предзагрузка E5-small модели..."
python -c "
from sentence_transformers import SentenceTransformer
print('Загрузка intfloat/multilingual-e5-small...')
model = SentenceTransformer('intfloat/multilingual-e5-small')
print('E5-small модель загружена')
"

# Скачивание NER модели - ВРЕМЕННО ОТКЛЮЧЕНО
# echo "Скачивание NER модели..."
# python -c "
# from huggingface_hub import snapshot_download
# import os
# 
# model_path = 'ml/nlp/models/rubio_ner_finetuned_74'
# if not os.path.exists(os.path.join(model_path, 'model.safetensors')):
#     print('Скачивание NER модели...')
#     snapshot_download(
#         repo_id='velto006/rubio-ner-finetuned-74',
#         local_dir=model_path,
#         local_dir_use_symlinks=False,
#         ignore_patterns=['*.pt', 'optimizer-*.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']
#     )
#     print('NER модель загружена')
# else:
#     print('NER модель уже есть в кеше')
# "

echo "Сборка завершена!"