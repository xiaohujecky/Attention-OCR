from data_util.voc_keys import alphabet
alphabet=[e.encode('utf-8') for e in alphabet]
alphabet2idx_map={}
alphabet2lex_map={}
for a_idx, lex in enumerate(alphabet):
    alphabet2idx_map[lex] = a_idx + 4
    alphabet2lex_map['%d'%(a_idx + 4)] = lex

alphabet2idx_map['BOS'] = 1
alphabet2idx_map['EOS'] = 2
alphabet2idx_map['UNK'] = 3
alphabet2lex_map['1'] = 'BOS'
alphabet2lex_map['2'] = 'EOS'
alphabet2lex_map['3'] = 'UNK'

print len(alphabet2lex_map)
#with open('/data/OCR/create_words_on_img_html/result/xuexin_words_pildraw_atribute_train_v6/atribuate_words/atribuate_words_001773666.txt') as f:
#    text=f.read()
#    print text.strip()
