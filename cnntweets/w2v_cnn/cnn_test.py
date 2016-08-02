#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import cnn_data_helpers
from text_cnn import TextCNN

from word2vecReader import Word2Vec
import time
import gc

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 100000, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
max_len = 60



class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):

        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def load_w2v():
    model_path = '/Users/bong/works/data/word2vec_twitter_model/word2vec_twitter_model.bin'
    with Timer("load w2v"):
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

    return model


w2vmodel = load_w2v()
# x_train, y_train = cnn_data_helpers.load_data('trn',w2vmodel , max_len)
# x_dev, y_dev = cnn_data_helpers.load_data('dev', w2vmodel, max_len)
x_test, y_test  = cnn_data_helpers.load_data('tst', w2vmodel, max_len)
del(w2vmodel)
gc.collect()


# savepath = '/Users/bong/works/tfen/w2v_cnn/runs/backup/checkpoints/model-2900'

savepath = '/Users/bong/works/tfen/w2v_cnn/runs/1464326614/checkpoints/model-6700'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_test.shape[1],
            num_classes=3,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)


        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,savepath)



        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)

            print("loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))




        print("\nTest:")
        test_step(x_test, y_test)
        print("")

