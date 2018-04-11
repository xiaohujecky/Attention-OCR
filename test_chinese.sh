#!/usr/bin/env bash

python src/launcher.py \
	--phase=test \
	--data-path=./2018-01-20.imshow.label \
	--data-base-dir=/ \
	--log-path=log_01_16_test.txt \
	--attn-num-hidden 256 \
	--batch-size 64 \
	--model-dir=model_chinese \
	--load-model \
	--num-epoch=3 \
	--gpu-id=1 \
	--output-dir=model_chinese_testout_xuexin/ \
	--use-gru \
    --target-embedding-size=20
	
#--data-path=/data/OCR/create_words_on_img_html/result/lmdb/xuexin_pil_atribute/valid.txt \
