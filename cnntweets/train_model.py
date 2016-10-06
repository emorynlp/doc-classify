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

import datetime
from utils import cnn_data_helpers
from utils.cnn_data_helpers import load_lexicon_unigram, load_w2v
from utils.butils import Timer
from cnn_models.w2v_lex_cnn import W2V_LEX_CNN, W2V_LEX_CNN_CONCAT, W2V_LEX_CNN_CONCAT_A2V
from cnn_models.w2v_cnn import W2V_CNN
from cnn_models.preattention_cnn import TextCNNPreAttention, TextCNNPreAttentionBias
from cnn_models.preattention_cnn import TextAttention2Vec, TextAttention2VecIndividual
from cnn_models.preattention_cnn import TextAttention2VecIndividualBias
from cnn_models.preattention_cnn import TextAttention2VecIndividualW2v, TextAttention2VecIndividualLex
from cnn_models.preattention_cnn import TextCNNAttention2VecIndividual
from cnn_models.preattention_cnn import TextCNNAttention2VecIndividualW2v, TextCNNAttention2VecIndividualLex
from cnn_models.multi_channel import W2V_LEX_CNN_MC, W2V_LEX_CNN_MC_A2V



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





def run_train(w2vsource, w2vdim, w2vnumfilters, lexdim, lexnumfilters, randomseed, model_name, is_expanded,
              attention_depth_w2v, attention_depth_lex, num_epochs, l2_reg_lambda, l1_reg_lambda, simple_run=True):
    if simple_run == True:
        print '======================================[simple_run]======================================'

    max_len = 60
    norm_model = []

    rt_list = ['w2vrt', 'w2vlexrt', 'attrt', 'attbrt', 'a2vrt', 'a2vindrt', 'a2vindbrt', 'a2vindw2vrt',
               'a2vindlexrt', 'cnna2vindrt', 'cnna2vindw2vrt', 'cnna2vindlexrt', 'cnnmcrt', 'w2vlexcrt',
               'w2vlexca2vrt', 'cnnmca2vrt']

    multichannel = False
    if model_name == 'cnnmc' or model_name == 'cnnmcrt' or model_name == 'cnnmca2v' or model_name == 'cnnmca2vrt':
        multichannel = True

    multichannel_a2v = False
    if model_name == 'cnnmca2v' or model_name == 'cnnmca2vrt':
        multichannel_a2v = True

    rt_data = False
    if model_name in rt_list:
        rt_data = True

    with Timer("lex"):
        if is_expanded == 0:
            print 'old way of loading lexicon'
            norm_model, raw_model = load_lexicon_unigram(lexdim)
            # with open('../data/lexicon_data/lex15.pickle', 'rb') as handle:
            #     norm_model = pickle.load(handle)

        else:
            print 'new way of loading lexicon'
            default_vector_dic = {'EverythingUnigramsPMIHS': [0],
                                  'HS-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                                  'Maxdiff-Twitter-Lexicon_0to1': [0.50403226],
                                  'S140-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                                  'unigrams-pmilexicon': [0, 0, 0],
                                  'unigrams-pmilexicon_sentiment_140': [0, 0, 0],
                                  'BL': [0]}

            lexfile_list = ['EverythingUnigramsPMIHS.pickle',
                            'HS-AFFLEX-NEGLEX-unigrams.pickle',
                            'Maxdiff-Twitter-Lexicon_0to1.pickle',
                            'S140-AFFLEX-NEGLEX-unigrams.pickle',
                            'unigrams-pmilexicon.pickle',
                            'unigrams-pmilexicon_sentiment_140.pickle',
                            'BL.pickle']

            for idx, lexfile in enumerate(lexfile_list):
                if is_expanded == 1234567:  # expand all
                    # fname = '../data/le/exp_compact.%s' % lexfile
                    # print 'expanded lexicon for exp_compact.%s' % lexfile
                    fname = '../data/le/exp_1.1.%s' % lexfile
                    print 'expanded lexicon for exp_1.1.%s' % lexfile


                elif is_expanded - 1 == idx:
                    # fname = '../data/le/exp_%s' % lexfile
                    # print 'expanded lexicon for exp_%s' % lexfile
                    fname = '../data/le/exp_compact.%s' % lexfile
                    print 'expanded lexicon for exp_compact.%s' % lexfile
                    # fname = '../data/le/exp_1.1.%s' % lexfile
                    # print 'expanded lexicon for exp_1.1.%s' % lexfile

                else:
                    fname = '../data/le/%s' % lexfile
                    print 'default lexicon for %s' % lexfile

                if is_expanded == 8:
                    fname = '../data/le/new/%s' % lexfile
                    print 'new default lexicon for %s' % lexfile

                with open(fname, 'rb') as handle:
                    each_model = pickle.load(handle)
                    default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
                    each_model["<PAD/>"] = default_vector
                    norm_model.append(each_model)

    with Timer("w2v"):
        w2vmodel = load_w2v(w2vdim, simple_run=simple_run, source=w2vsource)
        # if w2vsource == "twitter":
        #     w2vmodel = load_w2v(w2vdim, simple_run=simple_run, source=w2vsource)
        # else:
        #     w2vmodel = load_w2v(w2vdim, simple_run=simple_run, source="amazon")



    unigram_lexicon_model = norm_model
    # unigram_lexicon_model = raw_model

    if simple_run:
        if multichannel_a2v is True or multichannel is True:
            x_train, y_train, x_lex_train, x_fat_train = \
                cnn_data_helpers.load_data('trn_sample', w2vmodel, unigram_lexicon_model,
                                                                       max_len, multichannel=multichannel)
            x_dev, y_dev, x_lex_dev, x_fat_dev = \
                cnn_data_helpers.load_data('dev_sample', w2vmodel, unigram_lexicon_model, max_len,
                                                                 multichannel=multichannel)
            x_test, y_test, x_lex_test, x_fat_test = \
                cnn_data_helpers.load_data('tst_sample', w2vmodel, unigram_lexicon_model, max_len,
                                                                    multichannel=multichannel)
        else:
            x_train, y_train, x_lex_train, _ = cnn_data_helpers.load_data('trn_sample', w2vmodel, unigram_lexicon_model,
                                                                       max_len, multichannel=multichannel)
            x_dev, y_dev, x_lex_dev, _ = cnn_data_helpers.load_data('dev_sample', w2vmodel, unigram_lexicon_model, max_len,
                                                                 multichannel = multichannel)
            x_test, y_test, x_lex_test, _ = cnn_data_helpers.load_data('tst_sample', w2vmodel, unigram_lexicon_model, max_len,
                                                                    multichannel=multichannel)

    else:
        if multichannel_a2v is True or multichannel is True:
            x_train, y_train, x_lex_train, x_fat_train = \
                cnn_data_helpers.load_data('trn', w2vmodel, unigram_lexicon_model, max_len,
                                           rottenTomato=rt_data, multichannel=multichannel)
            x_dev, y_dev, x_lex_dev, x_fat_dev = \
                cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len,
                                           rottenTomato=rt_data, multichannel=multichannel)
            x_test, y_test, x_lex_test, x_fat_test = \
                cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len,
                                           rottenTomato=rt_data, multichannel=multichannel)
        else:
            x_train, y_train, x_lex_train, _ = cnn_data_helpers.load_data('trn', w2vmodel, unigram_lexicon_model, max_len,
                                                                       rottenTomato=rt_data, multichannel=multichannel)
            x_dev, y_dev, x_lex_dev, _ = cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len,
                                                                 rottenTomato=rt_data, multichannel=multichannel)
            x_test, y_test, x_lex_test, _ = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len,
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
            if model_name in rt_list:
                num_classes = 5

            if model_name == 'w2v' or model_name == 'w2vrt':
                cnn = W2V_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'w2vlex' or model_name == 'w2vlexrt':
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

            elif model_name == 'att' or model_name == 'attrt':
                cnn = TextCNNPreAttention(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'attb' or model_name == 'attbrt':
                cnn = TextCNNPreAttentionBias(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'a2v' or model_name == 'a2vrt':
                cnn = TextAttention2Vec(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'a2vind' or model_name == 'a2vindrt':
                cnn = TextAttention2VecIndividual(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'a2vindb' or model_name == 'a2vindbrt':
                cnn = TextAttention2VecIndividualBias(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'a2vindw2v' or model_name == 'a2vindw2vrt':
                cnn = TextAttention2VecIndividualW2v(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'a2vindlex' or model_name == 'a2vindw2vrt':
                cnn = TextAttention2VecIndividualLex(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'cnna2vind' or model_name == 'cnna2vindrt':
                cnn = TextCNNAttention2VecIndividual(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'cnna2vindw2v' or model_name == 'cnna2vindw2vrt':
                cnn = TextCNNAttention2VecIndividualW2v(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'cnna2vindlex' or model_name == 'cnna2vindlexrt':
                cnn = TextCNNAttention2VecIndividualLex(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)


            elif model_name =='cnnmc' or model_name =='cnnmcrt':
                cnn = W2V_LEX_CNN_MC(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'w2vlexc' or model_name == 'w2vlexcrt':
                cnn = W2V_LEX_CNN_CONCAT(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'w2vlexca2v' or model_name == 'w2vlexca2vrt':
                cnn = W2V_LEX_CNN_CONCAT_A2V(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=attention_depth_w2v,
                    attention_depth_lex=attention_depth_lex,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            elif model_name == 'cnnmca2v' or model_name == 'cnnmca2vrt':
                cnn = W2V_LEX_CNN_MC_A2V(
                    sequence_length=x_train.shape[1],
                    num_classes=num_classes,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    attention_depth_w2v=50,
                    attention_depth_lex=20,
                    l2_reg_lambda=l2_reg_lambda,
                    l1_reg_lambda=l1_reg_lambda)

            else: # default is w2vlex
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

            def test_step(x_batch, y_batch, x_batch_lex=None, x_batch_fat=None, writer=None, score_type='f1', multichannel=False):
                """
                Evaluates model on a test set
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
                      format("TEST", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                if writer:
                    writer.add_summary(summaries, step)

                if score_type == 'f1':
                    return avg_f1
                else:
                    return accuracy

            # Generate batches
            if multichannel_a2v is True or multichannel is True:
                batches = cnn_data_helpers.batch_iter(
                    list(zip(x_train, y_train, x_lex_train, x_fat_train)), FLAGS.batch_size, num_epochs)
            else:
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
                    # train_step(x_batch, y_batch, x_batch_lex)


                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation:")
                    if rt_data == True:
                        score_type = 'acc'
                    else:
                        score_type = 'f1'

                    if model_name == 'w2v' or model_name == 'w2vrt':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type=score_type)
                        curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer, score_type=score_type)

                    else:
                        if multichannel_a2v is True:
                            curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, x_fat_dev, writer=dev_summary_writer,
                                                    score_type=score_type, multichannel=multichannel)
                            curr_af1_tst = test_step(x_test, y_test, x_lex_test, x_fat_test, writer=test_summary_writer,
                                                     score_type=score_type, multichannel=multichannel)

                        elif multichannel is True:
                            curr_af1_dev = dev_step(x_dev, y_dev, x_batch_lex=None, x_batch_fat=x_fat_dev,
                                                    writer=dev_summary_writer,
                                                    score_type=score_type, multichannel=multichannel)
                            curr_af1_tst = test_step(x_test, y_test, x_batch_lex=None, x_batch_fat=x_fat_test,
                                                     writer=test_summary_writer,
                                                     score_type=score_type, multichannel=multichannel)
                        else:
                            curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer,
                                                    score_type=score_type, multichannel=multichannel)
                            curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer,
                                                     score_type = score_type, multichannel=multichannel)


                    # if model_name == 'w2v':
                    #     curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    #     # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     # print("Saved model checkpoint to {}\n".format(path))
                    #
                    #     curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer)
                    #     # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     # print("Saved model checkpoint to {}\n".format(path))
                    #
                    # elif model_name == 'w2vrt':
                    #     curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type='acc')
                    #     curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer, score_type='acc')
                    #
                    # elif model_name == 'w2vlexrt':
                    #     curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer, score_type='acc')
                    #     curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer,
                    #                              score_type='acc')
                    # else:
                    #     curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer)
                    #     # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     # print("Saved model checkpoint to {}\n".format(path))
                    #
                    #     curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer)
                    #     # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     # print("Saved model checkpoint to {}\n".format(path))

                    if curr_af1_dev > max_af1_dev:
                        max_af1_dev = curr_af1_dev
                        index_at_max_af1_dev = current_step
                        af1_tst_at_max_af1_dev = curr_af1_tst

                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                    if rt_data == True:
                        print 'Status: [%d] Max Acc for dev (%f), Max Acc for tst (%f)\n' % (
                            index_at_max_af1_dev, max_af1_dev*100, af1_tst_at_max_af1_dev*100)
                    else:
                        print 'Status: [%d] Max f1 for dev (%f), Max f1 for tst (%f)\n' % (
                            index_at_max_af1_dev, max_af1_dev, af1_tst_at_max_af1_dev)


                    sys.stdout.flush()

                    # if current_step % FLAGS.test_every == 0:
                    #     print("\nTest:")
                    #     if test_step(x_test, y_test, writer=test_summary_writer) is True:
                    #         path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #         print("Saved model checkpoint to {}\n".format(path))
                    #     print("")

                    # if current_step % FLAGS.checkpoint_every == 0:
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--w2vsource', default='twitter', choices=['twitter', 'amazon'], type=str)
    parser.add_argument('--w2vdim', default=400, type=int)
    parser.add_argument('--w2vnumfilters', default=64, type=int)
    parser.add_argument('--lexdim', default=15, type=int)
    parser.add_argument('--lexnumfilters', default=9, type=int)
    parser.add_argument('--randomseed', default=1, type=int)
    parser.add_argument('--model', default='w2v', choices=['w2v', 'w2vrt', 'w2vlex', 'w2vlexrt',
                                                             'att', 'attrt', 'attb', 'attbrt', 'a2v', 'a2vrt',
                                                             'a2vind', 'a2vindrt', 'a2vindb', 'a2vindbrt',
                                                             'a2vindw2v', 'a2vindw2vrt', 'a2vindlex', 'a2vindlexrt',
                                                             'cnna2vind', 'cnna2vindrt',
                                                             'cnna2vindw2v', 'cnna2vindw2vrt',
                                                             'cnna2vindlex', 'cnna2vindlexrt',
                                                             'cnnmc','cnnmcrt',
                                                             'w2vlexc', 'w2vlexcrt',
                                                             'w2vlexca2v', 'w2vlexca2vrt',
                                                             'cnnmca2v', 'cnnmca2vrt'
                                                              ],
                        type=str)  # w2v, w2vlex, attention
    parser.add_argument('--expanded', default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 1234567], type=int)
    parser.add_argument('--attdepthw2v', default=50, type=int)
    parser.add_argument('--attdepthlex', default=20, type=int)
    parser.add_argument('--num_epochs', default=25, type=int)
    parser.add_argument('--l2_reg_lambda', default=2.0, type=float)
    parser.add_argument('--l1_reg_lambda', default=0.0, type=float)



    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])

    print 'ADDITIONAL PARAMETER\n w2vsource: %s\n w2vdim: %d\n w2vnumfilters: %d\n lexdim: %d\n lexnumfilters: %d\n ' \
          'randomseed: %d\n model_name: %s\n expanded: %d\n attdepthw2v: %s\n attdepthlex: %s\n num_epochs: %d\n' \
          'l2_reg_lambda: %f\n l2_reg_lambda: %f\n' \
          % (args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
             args.model, args.expanded, args.attdepthw2v, args.attdepthlex, args.num_epochs,
             args.l2_reg_lambda, args.l1_reg_lambda)

    run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
              args.model, args.expanded, args.attdepthw2v, args.attdepthlex, args.num_epochs,
              args.l2_reg_lambda, args.l1_reg_lambda,
              simple_run=False)
    # run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
    #           args.model, args.expanded, args.attdepthw2v, args.attdepthlex, args.num_epochs,
    #           args.l2_reg_lambda, args.l1_reg_lambda,
    #           simple_run=True)

