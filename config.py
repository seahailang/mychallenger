#!/usr/bin/env python
# encoding: utf-8


"""
@version: 0.0
@author: hailang
@Email: seahailang@gmail.com
@software: PyCharm Community Edition
@file: config.py
@time: 2017/9/20 16:06
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

''' model detail'''

tf.app.flags.DEFINE_integer('en_vocab_size',20000,
                            '''english vocabulary size''')
tf.app.flags.DEFINE_integer('en_embedded_size',200,
                          '''english embedded size''')
tf.app.flags.DEFINE_integer('en_max_length',20,
                           '''''')
tf.app.flags.DEFINE_integer('zh_vocab_size',10000,
                            '''english vocabulary size''')
tf.app.flags.DEFINE_integer('zh_embedded_size',200,
                          '''english embedded size''')
tf.app.flags.DEFINE_integer('zh_max_length',20,
                           '''''')
tf.app.flags.DEFINE_integer('batch_size',64,
                            '''batch size''')
tf.app.flags.DEFINE_integer('attention_size',100,
                           '''''')



'''train detail'''
tf.app.flags.DEFINE_float('learning_rate',0.01,
                          '''initial learning rate''')
tf.app.flags.DEFINE_integer('decay_step',1000,
                            '''decay step''')
tf.app.flags.DEFINE_float('decay_rate',0.99,
                          '''decay weight''')
tf.app.flags.DEFINE_float('max_gradient',1.00,
                          '''clipped max gradient''')


'''train flags'''
tf.app.flags.DEFINE_boolean('is_inference',False,
                            '''inference flag''')
tf.app.flags.DEFINE_boolean('is_train',True,
                            '''train flag''')



'''data dir'''
tf.app.flags.DEFINE_string("train_dir",'E:\\AI_Challenger\\ai_challenger_translation_train_20170904\\translation_train_data_20170904',
                           '''train_dir''')
tf.app.flags.DEFINE_string('en_vocab_dir','E:\\AI_Challenger\\my_challenge\\vocab.en',
                            '''en vocabulary dir''')
tf.app.flags.DEFINE_string('zh_vocab_dir','E:\\AI_Challenger\\my_challenge\\vocab.zh',
                            '''zh vocabulary dir''')
tf.app.flags.DEFINE_string('val_dir',
                           '''E:\AI_Challenger\\ai_challenger_translation_validation_20170912\\translation_validation_20170912''',
                           'validation dir')
tf.app.flags.DEFINE_string('test_dir',
                           'E:\\AI_Challenger\\ai_challenger_translation_test_a_20170923',
                           'test dir')

if __name__ == '__main__':
    pass