#! /usr/bin/env python
'''
 Copyright 2016, Emory University

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''
import os
import argparse
import tensorflow as tf
import numpy as np
from shutil import copyfile

import datetime
from utils import cnn_data_helpers
from utils.cnn_data_helpers import load_lexicon_unigram, load_w2v, load_w2v_withpath
from utils.butils import Timer
from cnn_models.w2v_lex_cnn import W2V_LEX_CNN
from cnn_models.w2v_cnn import W2V_CNN
import os.path
import utils.word2vecReaderUtils as utils



from utils.word2vecReader import Word2Vec
import time
import gc
import re
import sys
import pickle

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# os.system('cls' if os.name == 'nt' else 'clear')



# run_train(args.w2vnumfilters, args.lexnumfilters, args.randomseed,
#           args.num_epochs, args.l2_reg_lambda, args.l1_reg_lambda,
#           simple_run=False)
def run_train(w2v_path, trn_path, dev_path, model_path, lex_path_list, w2vnumfilters, lexnumfilters, randomseed,
              num_epochs, l2_reg_lambda, l1_reg_lambda, simple_run=True):
    if simple_run == True:
        print '======================================[simple_run]======================================'

    if len(lex_path_list)==0:
        model_name = 'w2v'
    else:
        model_name = 'w2vlex'

    best_model_path = None

    max_len = 60

    multichannel = False
    multichannel_a2v = False
    rt_data = False

    with utils.smart_open(w2v_path) as fin:
        header = utils.to_unicode(fin.readline())
        w2vdim = int(header.split(' ')[1].strip())


    with Timer("w2v"):
        w2vmodel = load_w2v_withpath(w2v_path)

    with Timer("lex"):
        print 'old way of loading lexicon'
        norm_model, raw_model = load_lexicon_unigram(lex_path_list)

    lexdim = 0
    for model_idx in range(len(norm_model)):
        lexdim += len(norm_model[model_idx].values()[0])

    unigram_lexicon_model = norm_model
    # unigram_lexicon_model = raw_model

    if simple_run:
        x_train, y_train, x_lex_train, _ = cnn_data_helpers.load_data('trn_sample', w2vmodel, unigram_lexicon_model,
                                                                   max_len, multichannel=multichannel)
        x_dev, y_dev, x_lex_dev, _ = cnn_data_helpers.load_data('dev_sample', w2vmodel, unigram_lexicon_model, max_len,
                                                             multichannel = multichannel)

    else:
        x_train, y_train, x_lex_train, _ = cnn_data_helpers.load_data(trn_path, w2vmodel, unigram_lexicon_model, max_len,
                                                                   rottenTomato=rt_data, multichannel=multichannel)
        x_dev, y_dev, x_lex_dev, _ = cnn_data_helpers.load_data(dev_path, w2vmodel, unigram_lexicon_model, max_len,
                                                             rottenTomato=rt_data, multichannel=multichannel)

    del (w2vmodel)
    del (norm_model)
    # del(raw_model)
    gc.collect()

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # Training
    # ==================================================
    if randomseed > 0:
        tf.set_random_seed(randomseed+10)
    with tf.Graph().as_default():
        max_af1_dev = 0
        index_at_max_af1_dev = 0
        af1_tst_at_max_af1_dev = 0

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if randomseed > 0:
                tf.set_random_seed(randomseed)

            num_classes = 3

            if model_name == 'w2v':
                cnn = W2V_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            else:
                cnn = W2V_LEX_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
            f1_summary = tf.scalar_summary("avg_f1", cnn.avg_f1)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, f1_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary, f1_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph_def)

            # Test summaries
            test_summary_op = tf.merge_summary([loss_summary, acc_summary, f1_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.train.SummaryWriter(test_summary_dir, sess.graph_def)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch, x_batch_lex=None, x_batch_fat=None, multichannel=False):
                """
                A single training step
                """
                if x_batch_fat is not None:
                    if x_batch_lex is None:
                        feed_dict = {
                            cnn.input_x: x_batch_fat,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }
                    else:
                        feed_dict = {
                            cnn.input_x_2c: x_batch_fat,
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.input_x_lexicon: x_batch_lex,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }

                else:
                    if x_batch_lex is None:
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }
                    else:
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            # lexicon
                            cnn.input_x_lexicon: x_batch_lex,
                            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                        }

                _, step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
                     cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                #      format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, x_batch_lex=None, x_batch_fat=None, writer=None, score_type='f1', multichannel=False):
                """
                Evaluates model on a dev set
                """
                if x_batch_fat is not None:
                    if x_batch_lex is None:
                        feed_dict = {
                            cnn.input_x: x_batch_fat,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0
                        }
                    else:
                        feed_dict = {
                            cnn.input_x_2c: x_batch_fat,
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.input_x_lexicon: x_batch_lex,
                            cnn.dropout_keep_prob: 1.0
                        }

                else:
                    if x_batch_lex is None:
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            cnn.dropout_keep_prob: 1.0
                        }
                    else:
                        feed_dict = {
                            cnn.input_x: x_batch,
                            cnn.input_y: y_batch,
                            # lexicon
                            cnn.input_x_lexicon: x_batch_lex,
                            cnn.dropout_keep_prob: 1.0
                        }

                step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy,
                     cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{} : {} step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                      format("DEV", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                if writer:
                    writer.add_summary(summaries, step)

                if score_type == 'f1':
                    return avg_f1
                else:
                    return accuracy

            # Generate batches
            batches = cnn_data_helpers.batch_iter(
                list(zip(x_train, y_train, x_lex_train)), FLAGS.batch_size, num_epochs)


            # Training loop. For each batch...
            for batch in batches:
                if multichannel_a2v is True or multichannel is True:
                    x_batch, y_batch, x_batch_lex, x_batch_fat = zip(*batch)
                else:
                    x_batch, y_batch, x_batch_lex = zip(*batch)

                if model_name == 'w2v' or model_name == 'w2vrt':
                    train_step(x_batch, y_batch)

                else:
                    if multichannel_a2v is True:
                        train_step(x_batch, y_batch, x_batch_lex, x_batch_fat)
                    elif multichannel is True:
                        train_step(x_batch, y_batch, x_batch_lex=None, x_batch_fat=x_batch_fat,
                                   multichannel=multichannel)
                    else:
                        train_step(x_batch, y_batch, x_batch_lex, multichannel=multichannel)

                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation:")
                    if rt_data == True:
                        score_type = 'acc'
                    else:
                        score_type = 'f1'

                    if model_name == 'w2v' or model_name == 'w2vrt':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type=score_type)

                    else:
                        curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer,
                                                score_type=score_type, multichannel=multichannel)

                    if curr_af1_dev > max_af1_dev:
                        max_af1_dev = curr_af1_dev
                        index_at_max_af1_dev = current_step

                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        best_model_path = path
                        print("Saved model checkpoint to {}\n".format(path))
                        copyfile(best_model_path, model_path)

                    if rt_data == True:
                        print 'Status: [%d] Max Acc for dev (%f)\n' % (
                            index_at_max_af1_dev, max_af1_dev*100)
                    else:
                        print 'Status: [%d] Max f1 for dev (%f)\n' % (
                            index_at_max_af1_dev, max_af1_dev)

                    sys.stdout.flush()





def get_lex_file_list(lexfile_path):
    lex_file_list = []
    with open(lexfile_path, 'rt') as handle:
        for line in handle.readlines():
            path = line.strip()

            if os.path.isfile(path):
                lex_file_list.append(path)
            else:
                print 'wrong file name(s) in the lex_config.txt\n%s' % path
                return None

    return lex_file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # python train_model.py - v w2v-400.bin - t train_data - d dev_data - l lex_config.txt - m model_file
    parser.add_argument('-v', default='../data/emory_w2v/w2v-50.bin', type=str) # w2v-400.bin
    parser.add_argument('-t', default='./trn', type=str) # train_data
    parser.add_argument('-d', default='./dev', type=str) # dev_data
    parser.add_argument('-l', default='./lex_config.txt', type=str) # lex_config.txt
    parser.add_argument('-m', default='./model_test', type=str) # model_file

    parser.add_argument('-w2vnumfilters', default=64, type=int)
    parser.add_argument('-lexnumfilters', default=9, type=int)
    parser.add_argument('-randomseed', default=1, type=int)
    parser.add_argument('-num_epochs', default=25, type=int)
    parser.add_argument('-l2_reg_lambda', default=2.0, type=float)
    parser.add_argument('-l1_reg_lambda', default=0.0, type=float)


    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])

    print 'ADDITIONAL PARAMETER\n w2vnumfilters: %d\n lexnumfilters: %d\n ' \
          'randomseed: %d\n num_epochs: %d\n' \
          'l2_reg_lambda: %f\n l2_reg_lambda: %f\n' \
          % (args.w2vnumfilters, args.lexnumfilters, args.randomseed,
             args.num_epochs, args.l2_reg_lambda, args.l1_reg_lambda)

    lex_list = get_lex_file_list(args.l)

    if not os.path.isfile(args.v):
        print 'wrong file name for the w2v binary\n%s' % args.v
        exit()

    if not os.path.isfile(args.t):
        print 'wrong trn file name\n%s' % args.t
        exit()

    if not os.path.isfile(args.d):
        print 'wrong dev file name\n%s' % args.d
        exit()

    if lex_list==None:
        exit()

    for l in lex_list:
        print l

    run_train(args.v, args.t, args.d, args.m, lex_list, args.w2vnumfilters, args.lexnumfilters, args.randomseed,
              args.num_epochs, args.l2_reg_lambda, args.l1_reg_lambda,
              simple_run=False)

    # run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
    #           args.model, args.expanded, args.attdepthw2v, args.attdepthlex, args.num_epochs,
    #           args.l2_reg_lambda, args.l1_reg_lambda,
    #           simple_run=True)

