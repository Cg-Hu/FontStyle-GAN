# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import argparse
import importlib
import sys
from importlib import reload # reload(module)重载模块。
# from imp import reload # reload(module)重载模块。

import numpy as np
import os
from PIL import Image #打开图片
from PIL import ImageDraw #创建图片
from PIL import ImageFont
import json
import collections
import matplotlib.pyplot as plt
reload(sys)
# sys.setdefaultencoding("utf-8")

CN_CHARSET = None
CN_T_CHARSET = None
JP_CHARSET = None
KR_CHARSET = None

DEFAULT_CHARSET = "/home/yxl/桌面/改动NICE-GAN/charset/cjk.json"


def load_global_charset():
    global CN_CHARSET, JP_CHARSET, KR_CHARSET, CN_T_CHARSET
    cjk = json.load(open(DEFAULT_CHARSET))
    CN_CHARSET = cjk["gbk"]
    JP_CHARSET = cjk["jp"]
    KR_CHARSET = cjk["kr"]
    CN_T_CHARSET = cjk["gb2312_t"]


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):  # 生成一个256*256的像素大小的字符，x偏移量20 y偏移量20
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    # 三通道改成单通道
    newimg = img.convert('L')

    draw = ImageDraw.Draw(newimg)
    draw.text((x_offset, y_offset), ch, (0), font=font)
    return newimg


def draw_exampleA(ch , src_font, canvas_size,x_offset, y_offset, filter_hashes):  # 这个函数返回用于生成一个含有两个字的512像素大的图片

    src_img = draw_single_char(ch, src_font, canvas_size, canvas_size / 4, canvas_size / 4)  # 这个是生成的目标文字的字符
    # src_hash = hash(src_img.tobytes())    #改动
    src_hash = hash(src_img.tobytes())    #改动

    if src_hash in filter_hashes: #如果没有字就不生成
        return None
    return src_img

def draw_exampleB(ch, dst_font, canvas_size,x_offset, y_offset, filter_hashes):   # 这个函数返回用于生成一个含有两个字的512像素大的图片
    dst_img = draw_single_char(ch, dst_font, canvas_size, canvas_size/4, canvas_size/4)  # 这个是生成的源文字的字符
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes: #如果没有字就不生成
        return None
    return dst_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    # 过滤到字符集中缺失的字  一共由26535个 double-click to see
    _charset = charset[:]
    np.random.shuffle(_charset) # 功能：生成随机字符
    sample = _charset[:2000] # 样本有两千？
    hash_count = collections.defaultdict(int) # 该函数返回一个类似字典的对象。defaultdict是Python内建字典类（dict）的一个子类
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1  # img.tobytes()将图片作为字节对象返回
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())  # #函数返回一个类似字典的对象？
    return [rh[0] for rh in recurring_hashes]  # 返回这个字符集

# 生成一千张带有源文字和目标字体的图片 1000由sample_count设置
def font2img(src, dst, charset, char_size, canvas_size,
             x_offset, y_offset,sample_count,label=0, filter_by_hash=True): # sample_count 指定抽样数量
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes= set()
    if filter_by_hash:
        filter_hashes= set(filter_recurring_hash(charset, src_font, canvas_size,x_offset, y_offset))  # 过滤哈希表的字符
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))


    count = 0
    trainA_font = {}  # 存储字符标签
    testA_font = {}
    for c in charset:
        if count == sample_count:
            break

        src = draw_exampleA(c, src_font, canvas_size,x_offset, y_offset, filter_hashes)
        if src:
            trainA_path = os.path.join('dataset/radical-1/trainA/', "%d_%04d.jpg" % (label, count))
            src.save(trainA_path)
            trainA_font[trainA_path] = c  # dict存储图片路径、字符

            testA_path = os.path.join('dataset/radical-1/testA/', "%d_%04d.jpg" % (label, count))
            src.save(testA_path)
            testA_font[testA_path] = c  # dict存储图片路径、字符

            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)

    # 写入txt文件
    with open('dataset/radical-1/trainA.txt', 'w') as f:
        for key, value in trainA_font.items():
            s = key + " " + value + "\n"
            f.write(s)

    with open('dataset/radical-1/testA.txt', 'w') as f:
        for key, value in testA_font.items():
            s = key + " " + value + "\n"
            f.write(s)
    count = 0
    trainB_font = {}  # 存储字符标签
    testB_font = {}
    for c in charset:
        if count == sample_count:
            break
        dst = draw_exampleB(c, dst_font, canvas_size,x_offset, y_offset, filter_hashes)
        if dst:
            trainB_path=os.path.join('dataset/radical-1/trainB/', "%d_%04d.jpg" % (label, count))
            dst.save(trainB_path)
            trainB_font[trainB_path] = c  # dict存储图片路径、字符

            testB_path=os.path.join('dataset/radical-1/testB/', "%d_%04d.jpg" % (label, count))
            dst.save(testB_path)
            testB_font[testB_path] = c  # dict存储图片路径、字符

            count += 1
            if count % 100 == 0:
                print("processed %d chars" % count)

            # 写入txt文件
    with open('dataset/radical-1/trainB.txt', 'w') as f:
        for key, value in trainB_font.items():
             s = key + " " + value + "\n"
             f.write(s)

    with open('dataset/radical-1/testB.txt', 'w') as f:
         for key, value in testB_font.items():
            s = key + " " + value + "\n"
            f.write(s)

load_global_charset()
parser = argparse.ArgumentParser(description='Convert font to images')
parser.add_argument('--src_font', dest='src_font', default='/home/yxl/zi2zi/fonts/MSYHBD.TTF',help='path of the source font')
parser.add_argument('--dst_font', dest='dst_font', default='/home/yxl/zi2zi/fonts/train_fonts/DFKai-SB.ttf ',help='path of the target font')
parser.add_argument('--filter', dest='filter', type=int, default=1, help='filter recurring characters')
parser.add_argument('--charset', dest='charset', type=str, default='CN',
                    help='charset, can be either: CN, JP, KR or a one line file')
parser.add_argument('--shuffle', dest='shuffle', type=int, default=0, help='shuffle a charset before processings')
parser.add_argument('--char_size', dest='char_size', type=int, default=120, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
parser.add_argument('--sample_count', dest='sample_count', type=int, default=1000, help='number of characters to draw')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples') #标注

args = parser.parse_args()

if __name__ == "__main__":
    if args.charset in ['CN', 'JP', 'KR', 'CN_T']:
        charset = locals().get("%s_CHARSET" % args.charset)
    else:
        charset = [c for c in open(args.charset).readline()[:-1].decode("utf-8")]
    if args.shuffle:
        np.random.shuffle(charset)   # 生成随机列表

    font2img(args.src_font,args.dst_font,charset, args.char_size,
             args.canvas_size,args.x_offset, args.y_offset,
             args.sample_count, args.label, args.filter)

   #python unpe-ttf2img.py --src_font=fonts/兰亭集序.ttf --dst_font=fonts/fan-song.ttf --charset=CN --sample_count=61000 --label=0 --filter=1 --shuffle=1