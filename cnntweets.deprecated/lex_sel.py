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
from utils.word2vecReader import Word2Vec
import time
import gc
import re
import sys

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
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



def load_w2v(w2vdim, simple_run = True):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
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

def run_train(w2vdim, w2vnumfilters, lexdim, lexnumfilters, randomseed, lex_col, simple_run = True):
    if simple_run == True:
        print '======================================[simple_run]======================================'


    max_len = 60

    with Timer("lex"):
        norm_model, raw_model = load_lexicon_unigram(lexdim)

    with Timer("w2v"):
        w2vmodel = load_w2v(w2vdim, simple_run=simple_run)

    unigram_lexicon_model = norm_model
    # unigram_lexicon_model = raw_model

    if simple_run:
        x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trn_sample', w2vmodel, unigram_lexicon_model,
                                                                   max_len)
        # print len(x_lex_train)
        # print len(x_lex_train[0])
        # print x_lex_train.shape
        # print x_lex_train[:, :, lex_col].shape
        x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('dev_sample', w2vmodel, unigram_lexicon_model, max_len)
        x_test, y_test, x_lex_test = cnn_data_helpers.load_data('tst_sample', w2vmodel, unigram_lexicon_model, max_len)



    else:
        x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trn', w2vmodel, unigram_lexicon_model, max_len)
        x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('dev', w2vmodel, unigram_lexicon_model, max_len)
        x_test, y_test, x_lex_test = cnn_data_helpers.load_data('tst', w2vmodel, unigram_lexicon_model, max_len)

        x_lex_train = x_lex_train[:, :, lex_col]
        x_lex_dev = x_lex_dev[:, :, lex_col]
        x_lex_test = x_lex_test[:, :, lex_col]

    del(w2vmodel)
    del(norm_model)
    del(raw_model)
    gc.collect()

    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Training
    # ==================================================
    if randomseed > 0:
        tf.set_random_seed(randomseed)
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
            cnn = W2V_LEX_CNN(
                sequence_length=x_train.shape[1],
                num_classes=3,
                embedding_size=w2vdim,
                embedding_size_lex=lexdim,
                num_filters_lex = lexnumfilters,
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
                #print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                #      format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
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
                print("{} : {} step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                      format("DEV", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                if writer:
                    writer.add_summary(summaries, step)

                return avg_f1

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
                print("{} : {} step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                      format("TEST", time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
                if writer:
                    writer.add_summary(summaries, step)

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
                    print("Evaluation:")
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
    parser.add_argument('--w2vdim', default=200, type=int)
    parser.add_argument('--w2vnumfilters', default=256, type=int)
    parser.add_argument('--lexdim', default=15, type=int)
    parser.add_argument('--lexnumfilters', default=9, type=int)
    parser.add_argument('--randomseed', default=7, type=int)
    parser.add_argument('--lexcol', nargs='+', type=int)

    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])


    if args.lexcol is not None:
        args.lexdim = len(args.lexcol)

    else:
        args.lexcol = range(args.lexdim)

    print 'ADDITIONAL PARAMETER\n w2vdim: %d\n w2vnumfilters: %d\n lexdim: %d\n lexnumfilters: %d\n randomseed: %d\n' % (
    args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed)

    print 'lexcol', args.lexcol

    run_train(args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed, args.lexcol, simple_run=False)
    # run_train(args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed, args.lexcol, simple_run=True)
