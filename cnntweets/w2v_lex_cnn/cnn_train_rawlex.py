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
import re
import sys

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_lex", 6, "Dimensionality of character embedding from LEXICON")
# tf.flags.DEFINE_integer("embedding_dim_lex", 14, "Dimensionality of character embedding from LEXICON")


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
    model_path = '../data/word2vec_twitter_model/word2vec_twitter_model.bin'
    with Timer("load w2v"):
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

    return model



def load_lexicon_unigram():
    if FLAGS.embedding_dim_lex==6:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'HS-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                              'S140-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'unigrams-pmilexicon.txt': [0],
                              'unigrams-pmilexicon_sentiment_140.txt': [0]}

    else:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0]}

    file_path = ["../data/lexicon_data/"+files for files in os.listdir("../data/lexicon_data") if files.endswith(".txt")]
    raw_model = [dict() for x in range(len(file_path))]
    norm_model = [dict() for x in range(len(file_path))]
    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../data/lexicon_data/", "")
        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector


        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''

                if FLAGS.embedding_dim_lex == 6:
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk

                        elif idx == 1:
                            data_vec.append(float(tk))

                        else:
                            continue

                else:
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk
                        else:
                            data_vec.append(float(tk))


                assert(key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
    # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)

        print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[1].keys()
        dictionary = dict(zip(keys, new_val))
        norm_model[index] = dictionary

        data_type = file_path[index].replace("../data/lexicon_data/", "")
        default_vector = default_vector_dic[data_type]

        dictionary["<PAD/>"] = default_vector
        # models.append(dictionary)

    return norm_model, raw_model


with Timer("lex"):
    norm_model, raw_model = load_lexicon_unigram()

with Timer("w2v"):
    w2vmodel = load_w2v()

unigram_lexicon_model = raw_model
# unigram_lexicon_model = raw_model

x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trn',w2vmodel, unigram_lexicon_model, max_len)
x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len)
x_test, y_test, x_lex_test  = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len)


# x_train, y_train = cnn_data_helpers.load_data('trn',w2vmodel , max_len)
# x_dev, y_dev = cnn_data_helpers.load_data('dev', w2vmodel, max_len)
# x_test, y_test  = cnn_data_helpers.load_data('tst', w2vmodel, max_len)
del(w2vmodel)
del(norm_model)
del(raw_model)
gc.collect()

print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    max_af1_dev = 0
    index_at_max_af1_dev = 0
    af1_tst_at_max_af1_dev = 0

    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=3,
            embedding_size=FLAGS.embedding_dim,
            embedding_size_lex=FLAGS.embedding_dim_lex,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
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

        def train_step(x_batch, y_batch, x_batch_lex):
            """
            A single training step
            """
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
            print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, x_batch_lex, writer=None):
            """
            Evaluates model on a dev set
            """
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
            print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            if writer:
                writer.add_summary(summaries, step)

            return avg_f1
            # if avg_f1>56:
            #     return True
            #
            # else:
            #     return False

        def test_step(x_batch, y_batch, x_batch_lex, writer=None):
            """
            Evaluates model on a test set
            """
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
            print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            if writer:
                writer.add_summary(summaries, step)

            # if avg_f1 > 56:
            #     return True
            #
            # else:
            #     return False

            return avg_f1

        # Generate batches
        batches = cnn_data_helpers.batch_iter(
            list(zip(x_train, y_train, x_lex_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch, x_batch_lex = zip(*batch)
            train_step(x_batch, y_batch, x_batch_lex)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                curr_af1_dev = dev_step(x_dev, y_dev, x_lex_dev, writer=dev_summary_writer)
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    # print("Saved model checkpoint to {}\n".format(path))
                print("")

                print("\nTest:")
                curr_af1_tst = test_step(x_test, y_test, x_lex_test, writer=test_summary_writer)
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    # print("Saved model checkpoint to {}\n".format(path))
                print("")

                if curr_af1_dev > max_af1_dev:
                    max_af1_dev = curr_af1_dev
                    index_at_max_af1_dev = current_step
                    af1_tst_at_max_af1_dev = curr_af1_tst

                print '\nStatus:\n[%d] Max f1 for dev (%f), Max f1 for tst (%f)\n' % (
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
