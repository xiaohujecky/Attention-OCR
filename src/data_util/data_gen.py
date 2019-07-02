# -*- encoding: utf-8 -*-
__author__ = 'moonkey'

import os, sys, logging
import numpy as np
from PIL import Image
from collections import Counter
import pickle as cPickle
import random, math
from path import Path
from PIL import Image,ImageDraw,ImageFont
import cv2
reload(sys)
sys.setdefaultencoding('utf8')
from data_util.bucketdata import BucketData
#from data_util.voc_keys_7401 import alphabet
from data_util.voc_keys_8571 import alphabet
alphabet=[e.encode('utf-8') for e in alphabet]
alphabet2idx_map={}
alphabet2lex_map={}
for a_idx, lex in enumerate(alphabet):
    alphabet2idx_map[lex] = a_idx + 4
    alphabet2lex_map[u'%d'%(a_idx + 4)] = lex

alphabet2idx_map[u'BOS'] = 1
alphabet2idx_map[u'EOS'] = 2
alphabet2idx_map[u' '] = 3
alphabet2lex_map[u'1'] = u'BOS'
alphabet2lex_map[u'2'] = u'EOS'
alphabet2lex_map[u'3'] = u' '

DEBUG = False

class DataGenTextOnLine():
    def __init__(self):
        bg_imgs_path = 'data/bg_image_data.lst'
        self.bg_img_idx = 0
        self.fonts_files = Path('data/fonts/chinese_simplified/').files()
        random.shuffle(self.fonts_files)
        self.fonts_num = len(self.fonts_files)
        self.fonts_idx = 0
        self.font_sizes=[14,16,18,20,22,24,26,28,30,32,34,36,38,42,46,50,56,60]
        self.font_words_valid = {}
        for font_file in self.fonts_files:
            font_name = font_file.split('/')[-1]
            font_name_valid_file = 'data/fonts/chinese_simplified.validation/' + font_name.encode('utf-8') + '.lst'
            #logging.info('font_valid_file : {}'.format(font_name_valid_file))
            if font_name not in self.font_words_valid:
                words_valid = {}
                with open(font_name_valid_file, 'rb') as f:
                    lines=unicode(f.read(),'utf-8').split('\n')
                    for line in lines:
                        itms = line.split(u' ')
                        if len(itms) == 2:
                            words_valid[itms[0]] = int(itms[1])
                self.font_words_valid[font_name] = words_valid

        # Load images.
        with open(bg_imgs_path, 'r') as f:
            self.bg_imgs_id = [line.strip() for line in f.readlines()]
        self.bg_imgs_num = len(self.bg_imgs_id)

        self.lexcal_file = 'data/lexcal_data.txt'
        self.lexcal_file_opened = open(self.lexcal_file, 'r')
        self.lexcal_lists = [ unicode(na.strip(), 'utf-8') for na in self.lexcal_file_opened.readlines()]
        self.lexcal_lists_len = len(self.lexcal_lists)
        self.lexcal_lists_idx = 0
        self.imgs_name = ''
        self.imgs_idx = 0


    def gen_bg_img(self, height, width, r=255, g=255, b=255):
        img = np.ones((height, width, 3), np.uint8)
        val = random.randint(random.randint(0,127),255)
        img = img*val
        return img, val

    def gen_bg_img_from_file(self, height, width):
        bg_img_crop = None
        while bg_img_crop is None:
            if not self.bg_img_idx < self.bg_imgs_num:
                self.bg_img_idx = 0
            bg_img_name = self.bg_imgs_id[self.bg_img_idx]
            bg_img = cv2.imread(bg_img_name)
            self.bg_img_idx += 1
            if bg_img is not None:
                bg_img_h, bg_img_w = bg_img.shape[:2]
                # harianzen mirror
                bg_img_flip_v = bg_img[:,::-1]
                bg_img_extend = bg_img[:]
                flip_idx = 0
                while bg_img_w < width :
                    if flip_idx % 2 == 0:
                        bg_img_extend = np.concatenate((bg_img_extend, bg_img_flip_v), axis=1)
                    else:
                        bg_img_extend = np.concatenate((bg_img_extend, bg_img), axis=1)
                    flip_idx += 1
                    bg_img_h, bg_img_w = bg_img_extend.shape[:2]

                # vertical mirror
                bg_img_flip_h = bg_img_extend[::-1,:]
                bg_img_extend2 = bg_img_extend[:]
                flip_idx = 0
                while bg_img_h < height :
                    if flip_idx % 2 == 0:
                        bg_img_extend2 = np.concatenate((bg_img_extend2, bg_img_flip_h), axis=0)
                    else:
                        bg_img_extend2 = np.concatenate((bg_img_extend2, bg_img_extend), axis=0)
                    flip_idx += 1
                    bg_img_h, bg_img_w = bg_img_extend2.shape[:2]

                bg_img = bg_img_extend2
                bg_img_h, bg_img_w = bg_img.shape[:2]
                if bg_img_h > height and bg_img_w > width :
                    y_max = bg_img_h - height
                    x_max = bg_img_w - width
                    y = random.randint(0, y_max-1)
                    x = random.randint(0, x_max-1)
                    bg_img_crop = bg_img[ y:y+height, x:x+width, :]
                    try:
                    #if bg_img_crop is not None:
                        bg_pixel_value = int(np.mean(bg_img_crop))
                    #else:
                    except:
                        bg_pixel_value = 255

        return bg_img_crop, bg_pixel_value

    def put_text_on_img(self, text_org=''):
        text = text_org
        font_size = random.choice(self.font_sizes)

        # to see the fonts effects
        if self.fonts_idx >= self.fonts_num:
            self.fonts_idx = 0
        font_file = self.fonts_files[self.fonts_idx]
        self.fonts_idx += 1

        font_name = font_file.split('/')[-1]
        text_put=u''
        for w in text:
            if w in self.font_words_valid[font_name] and self.font_words_valid[font_name][w] == 1:
                text_put += w
        text_len = len(text_put)
        if not text_len >= 1:
            return None,u''
        #logging.info('font_file : {}, text_put : {}, text_len : {}'.format(font_file, text_put.encode('utf-8'), text_len))
        font = ImageFont.truetype(font_file, font_size)
        text_h = font_size
        text_w = text_h*text_len
        w_extend = min(int(text_w*0.2), 48)
        w_border = int(w_extend/2.0)
        h_border = min(int(text_h/4.0), 32)
        gen_img_h = text_h + 2*h_border
        gen_img_w = text_w + 2*w_border
        if random.randint(1,10) == 1:
            img, bg_pixel_val = self.gen_bg_img(gen_img_h, gen_img_w)
        else:
            img, bg_pixel_val = self.gen_bg_img_from_file(gen_img_h, gen_img_w)
        if img is None:
            img, bg_pixel_val = self.gen_bg_img(gen_img_h, gen_img_w)
        img=Image.fromarray(np.uint8(img))
        loc=(random.randint(0, max(w_border*2 - 1, 1)), random.randint(0, max(h_border*2 - 1, 1)))
        draw=ImageDraw.Draw(img)
        # black text
        rgb = random.randint(0, random.randint(127, 255))
        while math.fabs(rgb - bg_pixel_val) < 50 :
            if bg_pixel_val < 127:
                rgb=random.randint(bg_pixel_val+51, 255)
            else:
                rgb=random.randint(0, bg_pixel_val - 51)
        # wihte text
        #logging.info('text_put_type : {}'.format(type(text_put)))
        draw.text(loc, text_put, (rgb, rgb, rgb), font=font)
        #logging.info('font_file : {}, text_put : {}, text_len : {}'.format(font_file, text_put.encode('utf-8'), text_len))

        blur_kernel = [(3,3)]
        if random.randint(0,5) == 1:
            img = cv2.GaussianBlur(np.array(img), random.choice(blur_kernel), 0)
        else:
            img = np.array(img)
        return img, text_put

    def close_lexcal_file(self):
        self.lexcal_file_opened.close()
        self.lexcal_file_opened = None

    def gen_a_sample(self):
        lex_val=''
        img = None
        try:
            lex_val = self.lexcal_file_opened.readline()
            if lex_cal == '':
                self.close_lexcal_file()
                self.lexcal_file_opened = open(self.lexcal_file, 'r')
                lex_val = self.lexcal_file_opened.readline()
        except:
            self.close_lexcal_file()
            self.lexcal_file_opened = open(self.lexcal_file, 'r')
            lex_val = self.lexcal_file_opened.readline()
        if lex_val != '':
            self.imgs_name = "img_{:0>9}.png".format(self.imgs_idx)
            self.imgs_idx += 1
            try:
                img, lex_val_put = self.put_text_on_img(lex_val)
            except Exception as e:
                logging.warning('Warning: Did not create an image.{}'.format(e))
                return None, u'', u''
            return img, unicode(lex_val.strip(), 'utf-8'), self.imgs_name
        else:
            return None,'', ''

    def gen_a_sample_inall(self):
        if self.lexcal_lists_idx >= self.lexcal_lists_len:
            self.lexcal_lists_idx = 0
        lex_val= self.lexcal_lists[self.lexcal_lists_idx]
        check_loop = 0
        while not len(lex_val) > 0:
            self.lexcal_lists_idx += 1
            check_loop += 1
            if self.lexcal_lists_idx >= self.lexcal_lists_len:
                self.lexcal_lists_idx = 0
            lex_val= self.lexcal_lists[self.lexcal_lists_idx]
            logging.warning('Warning: drawing in the len(lex_val) == 0 loop, tried {} times.'.format(check_loop))
            if check_loop > 1000:
                break
        try:
            self.imgs_name = "img_{:0>9}.png".format(self.imgs_idx)
            self.imgs_idx += 1
            #logging.info('put_text_start : {}'.format(lex_val.encode('utf-8')))
            img, lex_val_put = self.put_text_on_img(lex_val)
            #logging.info('put_text_return : {}'.format(lex_val_put.encode('utf-8')))
            self.lexcal_lists_idx += 1
            #if isinstance(lex_val , str):
            #    lex_val = unicode(lex_val, 'utf-8')
            if DEBUG:
                def check_path(a_path):
                    if not os.path.exists(a_path):
                        os.makedirs(a_path)
                check_path('debug/img/')
                with open('debug/img_label.txt', 'a') as f:
                    f.write('img/{} {}\n'.format(self.imgs_name, lex_val_put.encode('utf-8')))
                cv2.imwrite('debug/img/{}'.format(self.imgs_name), img)
        except Exception as e:
            logging.warning('Warning: Did not create an image.{}'.format(e))
            return None, u'', u''

        return img, lex_val_put, self.imgs_name


class DataGen(object):
    GO = 1
    BOS = 1
    EOS = 2
    UNK = 3

    def __init__(self,
                 data_root, annotation_fn, train_sample_size=None,
                 evaluate = False,
                 valid_target_len = float('inf'),
                 img_width_range = (12, 320),
                 word_len = 60):
        """
        :param data_root:
        :param annotation_fn:
        :param lexicon_fn:
        :param img_width_range: only needed for training set
        :return:
        """

        img_height = 32
        self.data_root = data_root
        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            self.annotation_path = os.path.join(data_root, annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)), (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)), (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2)),
                                 (int(math.ceil(img_width_range[1] / 2)), word_len + 2)]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2), (int(108 / 4), 15 + 2),
                             (int(140 / 4), 17 + 2), (int(256 / 4), 20 + 2),
                             (int(math.ceil(img_width_range[1] / 4)), word_len + 2),
                             (int(math.ceil(img_width_range[1] / 2)), word_len + 2)]

        self.bucket_min_width, self.bucket_max_width = img_width_range
        self.image_height = img_height
        self.valid_target_len = valid_target_len
        self.train_sample_size = train_sample_size

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

        self.OnLineDataGen = DataGenTextOnLine()

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def get_target_voc_size(self):
        return max([int(idx)+1 for idx in alphabet2lex_map.keys()])

    def get_size(self):
        with open(self.annotation_path, 'r') as ann_file:
            return len(ann_file.readlines())

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            for l in lines:
                #img_path, lex = l.strip().split()
                l = unicode(l, 'utf-8')
                itms = l.strip().split(u' ')
                img_path = itms[0]
                lex = u' '.join([tt for tt in itms[1:]])
                try:
                    img_bw, word = self.read_data(img_path, lex)
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                        else:
                            pass
                            #assert False, 'no valid bucket of width {}, img {}'.format(width, l)
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    def gen_multi_source_data(self, batch_size):
        valid_target_len = self.valid_target_len
        with open(self.annotation_path, 'r') as ann_file:
            lines = ann_file.readlines()
            random.shuffle(lines)
            self.c_data_idx = 0
            self.c_data_num = len(lines)
            dataSourceLocal = 1
            for didx in range(self.train_sample_size):
                #img_path, lex = l.strip().split()
                def gen_local_data():
                    if self.c_data_idx >= self.c_data_num:
                        self.c_data_idx = 0
                    img_bw = None
                    word = u''
                    img_path = u''
                    def get_a_local_data():
                        l = unicode(lines[self.c_data_idx], 'utf-8')
                        itms = l.strip().split(u' ')
                        img_path = itms[0]
                        while img_path == u'':
                            self.c_data_idx += 1
                            if self.c_data_idx >= self.c_data_num:
                                self.c_data_idx = 0
                            l = unicode(lines[self.c_data_idx], 'utf-8')
                            itms = l.strip().split(u' ')
                            img_path = itms[0]
                        lex = u' '.join([tt for tt in itms[1:]])
                        img_bw, word = self.read_data(img_path, lex)
                        return img_bw, word, img_path
                    try:
                        img_bw, word, img_path = get_a_local_data()
                        self.c_data_idx += 1
                    except:
                        logging.warning("Warining: local data {} reading wrong.".format(img_path))
                        img_bw = None
                        word = u''
                        img_path = u''
                        self.c_data_idx += 1
                    return img_bw, word, img_path

                def gen_online_data():
                    np_img, lex, img_path = self.OnLineDataGen.gen_a_sample_inall()
                    try_times = 0
                    while np_img is None:
                        try_times += 1
                        np_img, lex, img_path = self.OnLineDataGen.gen_a_sample_inall()
                        if try_times > 20:
                            logging.info("Warining: Online data generator is wrong! Tried {} times.".format(try_times))
                            break
                    img_bw, word = self.cvt_data(np_img, lex)
                    return img_bw, word, img_path
                data_source_cur = 'random'
                try:
                    if dataSourceLocal == 1:
                        dataSourceLocal = random.randint(1,5)
                        img_bw, word, img_path = gen_local_data()
                        #logging.info('Warning:local source.')
                        data_source_cur = 'local'
                    else:
                        dataSourceLocal = random.randint(1,5)
                        img_bw, word, img_path = gen_online_data()
                        #logging.info('Warning:online data. idx : {}'.format(self.OnLineDataGen.lexcal_lists_idx))
                        data_source_cur = 'online'
                    if img_bw is None:
                        logging.warning("Waring : {} source data is None, img_path : {}".format(data_source_cur, img_path))
                        continue
                    if valid_target_len < float('inf'):
                        word = word[:valid_target_len + 1]
                    width = img_bw.shape[-1]

                    # TODO:resize if > 320
                    b_idx = min(width, self.bucket_max_width)
                    #logging.info("b_idx: {}.".format(b_idx))
                    #bs = self.bucket_data[b_idx].append(img_bw, word, os.path.join(self.data_root,img_path))
                    bs = self.bucket_data[b_idx].append(img_bw, word, img_path)
                    if bs >= batch_size:
                        b = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                        if b is not None:
                            yield b
                            dataSourceLocal = random.randint(1,5)
                        else:
                            pass
                            #assert False, 'no valid bucket of width {}, img {}'.format(width, l)
                except IOError:
                    pass # ignore error images
                    #with open('error_img.txt', 'a') as ef:
                    #    ef.write(img_path + '\n')
        self.clear()

    # DEPRECATED :
    def read_data_org(self, img_path, lex, dprecated=True):
        assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        for c in lex:
            assert 96 < ord(c) < 123 or 47 < ord(c) < 58
            word.append(
                ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3)
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


    def read_data(self, img_path, lex):
        #assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        #lex = unicode(lex, 'utf-8')
        #print "{} {}".format(lex.encode('utf-8'), len(lex))
        #assert 0 < len(lex) < self.bucket_specs[-1][1]
        if not len(lex) < self.bucket_specs[-1][1]:
            lex = lex[0:self.bucket_specs[-1][1]-1]
        for c in lex:
            utf_lex = c.encode('utf-8')
            if utf_lex not in alphabet2idx_map:
                logging.info("Warning: UNKNOW CHARACTER {}".format(utf_lex))
                utf_lex = u' '
            word.append(alphabet2idx_map[utf_lex])
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)
        #logging.info('Info:local source {}'.format(os.path.join(self.data_root, img_path)))

        return img_bw, word

    def cvt_data(self, np_img, lex):
        if np_img is not None:
            img = Image.fromarray(np.uint8(np_img))
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]
        else:
            return None,''

        # 'a':97, '0':48
        word = [self.GO]
       #lex = unicode(lex, 'utf-8')
       #print "{} {}".format(lex.encode('utf-8'), len(lex))
       #assert 0 < len(lex) < self.bucket_specs[-1][1]
        if not len(lex) < self.bucket_specs[-1][1]:
            lex = lex[0:self.bucket_specs[-1][1]-1]
        for c in lex:
            utf_lex = c.encode('utf-8')
            if utf_lex not in alphabet2idx_map:
                logging.info("Warning: UNKNOW CHARACTER {}".format(utf_lex))
                utf_lex = u' '
            word.append(alphabet2idx_map[utf_lex])
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word


    def read_data_chinese_labelfile(self, img_path, lex_file):
        #assert 0 < len(lex) < self.bucket_specs[-1][1]
        # L = R * 299/1000 + G * 587/1000 + B * 114/1000
        with open(os.path.join(self.data_root, img_path), 'rb') as img_file:
            img = Image.open(img_file)
            w, h = img.size
            aspect_ratio = float(w) / float(h)
            if aspect_ratio < float(self.bucket_min_width) / self.image_height:
                img = img.resize(
                    (self.bucket_min_width, self.image_height),
                    Image.ANTIALIAS)
            elif aspect_ratio > float(
                    self.bucket_max_width) / self.image_height:
                img = img.resize(
                    (self.bucket_max_width, self.image_height),
                    Image.ANTIALIAS)
            elif h != self.image_height:
                img = img.resize(
                    (int(aspect_ratio * self.image_height), self.image_height),
                    Image.ANTIALIAS)

            img_bw = img.convert('L')
            img_bw = np.asarray(img_bw, dtype=np.uint8)
            img_bw = img_bw[np.newaxis, :]

        # 'a':97, '0':48
        word = [self.GO]
        with open(lex_file, 'r') as f :
            lex = unicode(f.read().strip(), 'utf-8')
            #print "{} {}".format(lex.encode('utf-8'), len(lex))
            assert 0 < len(lex) < self.bucket_specs[-1][1]
            for c in lex:
                utf_lex = c.encode('utf-8')
                if utf_lex not in alphabet2idx_map:
                    print("WARNING : UNKNOW CHARACTER {}".format(utf_lex))
                    utf_lex = u'UNK'
                word.append(alphabet2idx_map[utf_lex])
        word.append(self.EOS)
        word = np.array(word, dtype=np.int32)
        # word = np.array( [self.GO] +
        # [ord(c) - 97 + 13 if ord(c) > 96 else ord(c) - 48 + 3
        # for c in lex] + [self.EOS], dtype=np.int32)

        return img_bw, word

def test_gen():
    print('testing gen_valid')
    # s_gen = EvalGen('../../data/evaluation_data/svt', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/iiit5k', 'test.txt')
    # s_gen = EvalGen('../../data/evaluation_data/icdar03', 'test.txt')
    s_gen = EvalGen('../../data/evaluation_data/icdar13', 'test.txt')
    count = 0
    for batch in s_gen.gen(1):
        count += 1
        print(str(batch['bucket_id']) + ' ' + str(batch['data'].shape[2:]))
        assert batch['data'].shape[2] == img_height
    print(count)


if __name__ == '__main__':
    test_gen()
