class config:
    ocr_model_path='model_chinese/translate.ckpt-543000'
    attn_num_layers=2
    attn_num_hidden=256
    batch_size=1
    use_gru=True
    gpu_id=1
    valid_target_len = float('inf')
    img_width_range=(12, 320)
    img_height=32
    word_len=30
    target_vocab_size=7335+2*26+10+4
    target_embedding_size=20
    visualize = True
