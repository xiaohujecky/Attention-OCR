import sys,os
import cv2
from predict import TextLineOCR


if __name__ == '__main__':
    test_lst = sys.argv[1]
    test_out = sys.argv[2]
    fout = open(test_out, 'w')
    imgs = []
    imgs_lst = []
    textline_ocr = TextLineOCR()
    with open(test_lst, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            img = cv2.imread(line)
            textline_res = textline_ocr.rec([img])
            for res in textline_res:
                res = [unicode(r, 'utf-8').encode('utf-8') for r in res]
                print("{} {}".format(line, ''.join(res)))
                fout.write("{} {}\n".format(line, ''.join(res)))
    fout.close()

