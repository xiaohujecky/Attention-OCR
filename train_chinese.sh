#!/usr/bin/env bash

nohup python src/launcher.py \
	--phase=train \
	--data-path=/data/OCR/create_words_on_img_html/result/lmdb/xuexin_pil_atribute/train.txt \
	--data-base-dir=/ \
	--log-path=chinese_v2.log \
	--attn-num-hidden 256 \
	--batch-size 64 \
	--model-dir=model_chinese \
	--initial-learning-rate=1.0 \
	--load-model \
	--num-epoch=200 \
	--gpu-id=0 \
	--use-gru \
	--steps-per-checkpoint=10000 \
    --target-embedding-size=20 | tee &2>1 &
