#!/bin/bash
# build.sh - скрипт сборки для Render

echo "Starting build..."

# Install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install huggingface-hub

# Collect static files
echo "Collecting static files..."
python backend/manage.py collectstatic --noinput

# Preload E5-small model - DISABLED
# echo "Preloading E5-small model..."
# python -c "
# from sentence_transformers import SentenceTransformer
# print('Loading intfloat/multilingual-e5-small...')
# model = SentenceTransformer('intfloat/multilingual-e5-small')
# print('E5-small model loaded')
# "

# Download NER model - DISABLED
# echo "Downloading NER model..."
# python -c "
# from huggingface_hub import snapshot_download
# import os
# 
# model_path = 'ml/nlp/models/rubio_ner_finetuned_74'
# if not os.path.exists(os.path.join(model_path, 'model.safetensors')):
#     print('Downloading NER model...')
#     snapshot_download(
#         repo_id='velto006/rubio-ner-finetuned-74',
#         local_dir=model_path,
#         local_dir_use_symlinks=False,
#         ignore_patterns=['*.pt', 'optimizer-*.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json']
#     )
#     print('NER model loaded')
# else:
#     print('NER model already cached')
# "

echo "Build finished!"