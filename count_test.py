#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: count_test.py
@time: 2017/9/29 14:19
"""
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import IPython
import os
import pynlpir
import re
import sys
import time
pynlpir.open()


def chinese_parse(s):
    return pynlpir.segment(s,pos_tagging=False)

def english_parse(s):
    pattern = re.compile('[a-zA-Z]+|\d+\.?\d+[a-zA-Z]*|[\-,\.\'\?\!"]')
    return re.findall(pattern,s)

def build_vocabulary(dir,analyzer):
    with open(dir,'r',encoding='utf-8') as file:
        word_counter = CountVectorizer(analyzer=analyzer,min_df =0.0001)
        word_counter.fit(file.readlines())
    return word_counter.vocabulary_


if __name__ == '__main__':
    dir = '../ai_challenger_translation_train_20170904/translation_train_data_20170904/train.zh.m'
    vocabulary = build_vocabulary(dir,chinese_parse)