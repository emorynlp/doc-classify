#! /usr/bin/env python
import os
import argparse
import tensorflow as tf
import numpy as np

import datetime
from utils import cnn_data_helpers
from utils.butils import Timer
from cnn_models.w2v_lex_cnn import W2V_LEX_CNN
from cnn_models.w2v_cnn import W2V_CNN
from cnn_models.attention_cnn import TextCNNAttention
from cnn_models.w2v_nonstatic import W2V_NONSTATIC
from utils.word2vecReader import Word2Vec
import time
import gc
import re
import sys
import pickle

from tensorflow.contrib import learn


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 75, "Number of training epochs (default: 200)")
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

def load_w2v2(w2vdim, simple_run = True, base_path = '../data/emory_w2v/'):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        model_path = base_path + 'w2v-%d.bin' % w2vdim
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

        return model


def load_w2v(w2vdim, simple_run = True, source = "twitter"):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        if source == "twitter":
            model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
        elif source == "amazon":
            model_path = '../data/emory_w2v/w2v-%d-%s.bin' % (w2vdim, source)

        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

        return model



def load_lexicon_unigram(lexdim):
    if lexdim==6:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'HS-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                              'S140-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'unigrams-pmilexicon.txt': [0],
                              'unigrams-pmilexicon_sentiment_140.txt': [0]}

    elif lexdim == 2:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0]}

    elif lexdim == 4:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0, 0, 0]}

    elif lexdim == 15:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}
    else:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}

    file_path = ["../data/lexicon_data/"+files for files in os.listdir("../data/lexicon_data") if files.endswith(".txt")]
    if lexdim == 2 or lexdim == 4:
        raw_model = [dict() for x in range(2)]
        norm_model = [dict() for x in range(2)]
        file_path = ['../data/lexicon_data/EverythingUnigramsPMIHS.txt', '../data/lexicon_data/unigrams-pmilexicon.txt']
    else:
        raw_model = [dict() for x in range(len(file_path))]
        norm_model = [dict() for x in range(len(file_path))]

    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../data/lexicon_data/", "")
        # if lexdim == 2 or lexdim == 4:
        #     if data_type not in ['EverythingUnigramsPMIHS.txt', 'unigrams-pmilexicon.txt']:
        #         continue

        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector

        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''

                if lexdim == 2 or lexdim == 6:
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk

                        elif idx == 1:
                            data_vec.append(float(tk))

                        else:
                            continue

                else: # 4 or 14
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk
                        else:
                            try:
                                data_vec.append(float(tk))
                            except:
                                pass


                assert(key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
    # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)


        #print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            #print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[index].keys()
        dictionary = dict(zip(keys, new_val))

        norm_model[index] = dictionary


    return norm_model, raw_model

def run_train(w2vsource, w2vdim, w2vnumfilters, lexdim, lexnumfilters, randomseed, model_name, trainable, is_expanded, simple_run = True):
    if simple_run == True:
        print '======================================[simple_run]======================================'


    max_len = 60
    norm_model = []

    if model_name != "nonstaticRT":
        with Timer("lex"):
            if is_expanded == 0:
                print 'old way of loading lexicon'
                norm_model, raw_model = load_lexicon_unigram(lexdim)


            else:
                print 'new way of loading lexicon'
                default_vector_dic = {'EverythingUnigramsPMIHS': [0],
                                      'HS-AFFLEX-NEGLEX-unigrams': [0, 0, 0],
                                      'Maxdiff-Twitter-Lexicon_0to1': [0.5],
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
                    if is_expanded-1 == idx:
                        fname = '../data/le/exp_%s' % lexfile
                        print 'expanded lexicon for %s' % lexfile

                    else:
                        fname = '../data/le/%s' % lexfile
                        print 'default lexicon for %s' % lexfile

                    with open(fname, 'rb') as handle:
                        each_model = pickle.load(handle)
                        default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
                        each_model["<PAD/>"] = default_vector
                        norm_model.append(each_model)


        with Timer("w2v"):
            if w2vsource == "twitter":
                w2vmodel = load_w2v(w2vdim, simple_run=simple_run)
            else:
                w2vmodel = load_w2v(w2vdim, simple_run=simple_run, source = "amazon")


        unigram_lexicon_model = norm_model

    # Training
    # ==================================================
    if randomseed > 0:
        tf.set_random_seed(randomseed)
    with tf.Graph().as_default():
        tf.set_random_seed(randomseed)
        max_af1_dev = 0
        index_at_max_af1_dev = 0
        af1_tst_at_max_af1_dev = 0

        x_text, y = cnn_data_helpers.load_data_nonstatic("everydata", rottenTomato=True)

        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        vocab_processor.fit_transform(x_text)
        total_vocab_size = len(vocab_processor.vocabulary_)


        if model_name == "w2vrt" or  model_name == "w2vrtlex":
            x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trn', w2vmodel, unigram_lexicon_model, max_len, True)
            x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len, True)
            x_test, y_test, x_lex_test = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len, True) 
        elif model_name == 'nonstaticRT':
            x_train, y_train = cnn_data_helpers.load_data_nonstatic("trn", rottenTomato=True)
            x_train = np.array(list(vocab_processor.fit_transform(x_train)))
            x_dev, y_dev = cnn_data_helpers.load_data_nonstatic("dev", rottenTomato=True)
            x_dev = np.array(list(vocab_processor.fit_transform(x_dev)))
            x_test, y_test = cnn_data_helpers.load_data_nonstatic("tst", rottenTomato=True)
            x_test = np.array(list(vocab_processor.fit_transform(x_test)))
        else:
            x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trn', w2vmodel, unigram_lexicon_model, max_len)
            x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len)
            x_test, y_test, x_lex_test = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len)

        if model_name != "nonstaticRT":
            del(w2vmodel)
            del(norm_model)
            # del(raw_model)
            gc.collect()

        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if randomseed > 0:
                tf.set_random_seed(randomseed)

            if model_name=='w2v':
                cnn = W2V_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=3,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )
            elif model_name == 'nonstaticRT':
                cnn = W2V_NONSTATIC(
                    sequence_length=x_train.shape[1],
                    num_classes=5,
                    vocab_size=len(vocab_processor.vocabulary_),
                    is_trainable=trainable,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )

            elif model_name=='w2vrt':
                cnn = W2V_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=5,
                    embedding_size=w2vdim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda
                )

            elif model_name=='w2vlex':
                cnn = W2V_LEX_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=3,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)
           
            elif model_name=='w2vrtlex':
                cnn = W2V_LEX_CNN(
                    sequence_length=x_train.shape[1],
                    num_classes=5,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

            else: # model_name == 'attention'
                cnn = TextCNNAttention(
                    sequence_length=x_train.shape[1],
                    num_classes=3,
                    embedding_size=w2vdim,
                    embedding_size_lex=lexdim,
                    num_filters_lex=lexnumfilters,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=w2vnumfilters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)


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
            the_base_path = '../data/emory_w2v/'
            if w2vsource == "twitter":
                the_model_path = the_base_path + 'w2v-%d.bin' % w2vdim
            elif w2vsource == "amazon":
                the_model_path = the_base_path + 'w2v-%d-%s.bin' % (w2vdim, w2vsource)

            print the_model_path
            if model_name == 'nonstaticRT':
                # initial matrix with random uniform
                initW = np.random.uniform(-0.25,0.25,(total_vocab_size, w2vdim))
                # load any vectors from the word2vec
                with Timer("w2v"):
                    print("Load word2vec file {} for NONSTATIC \n".format(the_model_path))
                    with open(the_model_path, "rb") as f:
                        header = f.readline()
                        vocab_size, layer1_size = map(int, header.split())
                        binary_len = np.dtype('float32').itemsize * layer1_size
                        for line in xrange(vocab_size):
                            word = []
                            while True:
                                ch = f.read(1)
                                if ch == ' ':
                                    word = ''.join(word)
                                    break
                                if ch != '\n':
                                    word.append(ch)   
                            idx = vocab_processor.vocabulary_.get(word)
                            if idx != 0:
                                #print str(idx) + " -> " + word
                                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32') 
                            else:
                                f.read(binary_len)    

                sess.run(cnn.W.assign(initW))

            def train_step(x_batch, y_batch, x_batch_lex=None):
                """
                A single training step
                """

                if x_batch_lex != None:
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        # lexicon
                        cnn.input_x_lexicon: x_batch_lex,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                else: 
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                _, step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
                     cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                #print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                #      format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, x_batch_lex=None, writer=None, score_type='f1'):
                """
                Evaluates model on a dev set
                """
                if x_batch_lex != None:
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        # lexicon
                        cnn.input_x_lexicon: x_batch_lex,
                        cnn.dropout_keep_prob: 1.0
                    }
                else: 
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
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

            def test_step(x_batch, y_batch, x_batch_lex=None, writer=None, score_type='f1'):
                """
                Evaluates model on a test set
                """
                if x_batch_lex != None:
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        # lexicon
                        cnn.input_x_lexicon: x_batch_lex,
                        cnn.dropout_keep_prob: 1.0
                    }
                else: 
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
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
            if model_name == 'nonstaticRT':
                batches = cnn_data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            else:
                batches = cnn_data_helpers.batch_iter(
                    list(zip(x_train, y_train, x_lex_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                if model_name == 'nonstaticRT':
                     x_batch, y_batch = zip(*batch)
                else:
                    x_batch, y_batch, x_batch_lex = zip(*batch)

                if model_name=='w2v' or model_name=='w2vrt' or model_name == 'nonstaticRT':
                    train_step(x_batch, y_batch)
                else:
                    train_step(x_batch, y_batch, x_batch_lex)


                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("Evaluation:")

                    if model_name == 'w2v':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        # print("Saved model checkpoint to {}\n".format(path))

                        curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer)
                        # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        # print("Saved model checkpoint to {}\n".format(path))

                    elif model_name == 'w2vrt' or model_name =='nonstaticRT':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type = 'acc')
                        curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer, score_type = 'acc')

                    elif model_name == 'w2vrtlex':
                        curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer, score_type = 'acc')
                        curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer, score_type = 'acc')
                    else:
                        curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer)
                            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            # print("Saved model checkpoint to {}\n".format(path))

                        curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer)
                            # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                            # print("Saved model checkpoint to {}\n".format(path))

                    if curr_af1_dev > max_af1_dev:
                        max_af1_dev = curr_af1_dev
                        index_at_max_af1_dev = current_step
                        af1_tst_at_max_af1_dev = curr_af1_tst

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
    parser.add_argument('--w2vsource', default='amazon', choices=['twitter','amazon'], type=str)
    parser.add_argument('--w2vdim', default=50, type=int)
    parser.add_argument('--w2vnumfilters', default=16, type=int)
    parser.add_argument('--lexdim', default=15, type=int)
    parser.add_argument('--lexnumfilters', default=9, type=int)
    parser.add_argument('--randomseed', default=7, type=int)
    parser.add_argument('--model', default='nonstaticRT', choices=['w2v', 'w2vrt', 'nonstaticRT', 'w2vlex', 'w2vrtlex', 'attention'], type=str) # w2v, w2vlex, attention
    parser.add_argument('--trainable', default='static', choices=['static', 'nonstatic'], type=str) # w2v, w2vlex, attention
    parser.add_argument('--expanded', default=0, choices=[0,1,2,3,4,5,6,7], type=int)

    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])

    print 'ADDITIONAL PARAMETER\n w2vsource: %s\n w2vdim: %d\n w2vnumfilters: %d\n lexdim: %d\n lexnumfilters: %d\n ' \
          'randomseed: %d\n model_name: %s\n model_name: %s\n expanded: %d' \
          % (args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
             args.model, args.trainable, args.expanded)

    run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
              args.model, args.trainable, args.expanded, simple_run=False)
