#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: val2record.py
@time: 2017/10/11 18:35
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import config
FLAGS = tf.app.flags.FLAGS

from xml.dom.minidom import parse
import xml.dom.minidom

def parse_xml(filename):
    data = []
    with open(filename,'r',encoding='utf-8') as file:
        for line in file.readlines()[4:-3]:
            line = line.split('>')[1][1:-6]
            data.append(line)
    return data



def main(argv=None):
    data = parse_xml(FLAGS.val_dir+'/valid.en-zh.en.sgm')
    print(data[-1])
    print(len(data))

if __name__ == '__main__':
    tf.app.run(main,None)