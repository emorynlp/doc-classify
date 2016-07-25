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

import csv
import re

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 400, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_dim_lex", 2, "Dimensionality of character embedding from LEXICON")
tf.flags.DEFINE_boolean("use_lexicon", False, "Set to use lexicon information. If False, set embedding_dim to 400")

tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev/test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate (default: 1e-3)")
tf.flags.DEFINE_boolean("random_seed", False, "Set to False to have same output everytime")
tf.flags.DEFINE_integer("seed_number", 0, "Seed number to use when using same random seed")
tf.flags.DEFINE_boolean("do_not_train", False, "FOR DEV PURPOSE")

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
max_len = 45



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
    model_path='./word2vec_twitter_model.bin'
    with Timer("loaded w2v"):
        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        #print("The vocabulary size is: " + str(len(model.vocab)))

    return model

def load_lexicon_unigram():
    file_path = ["./lexicon_data/"+files for files in os.listdir("./lexicon_data") if files.endswith(".txt")]
    model = [dict() for x in range(len(file_path))]
    with Timer("loaded unigram lexicon"):
        for index, each_model in enumerate(model):
            with open(file_path[index], 'r') as document:
                for line in document:
                    line_token = re.split(r'\t', line)
                    model[index][line_token[0]] = line_token[1]

    return [each_model for each_model in model]

unigram_lexicon_model = load_lexicon_unigram()
w2vmodel = load_w2v()

x_train, y_train, x_lex_train = cnn_data_helpers.load_data('trnTokenized.tsv',w2vmodel, unigram_lexicon_model,FLAGS.use_lexicon, max_len)
x_dev, y_dev, x_lex_dev = cnn_data_helpers.load_data('devTokenized.tsv', w2vmodel, unigram_lexicon_model, FLAGS.use_lexicon, max_len)
x_test, y_test, x_lex_test  = cnn_data_helpers.load_data('tstTokenized.tsv', w2vmodel, unigram_lexicon_model, FLAGS.use_lexicon, max_len)
del(w2vmodel)
gc.collect()

print("Train/Dev/Test split: {:d} / {:d} / {:d} \n".format(len(y_train), len(y_dev), len(y_test)))


# Training
# ==================================================
if not FLAGS.random_seed: tf.set_random_seed(FLAGS.seed_number)
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        if not FLAGS.random_seed: tf.set_random_seed(FLAGS.seed_number)
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=3,
            embedding_size=FLAGS.embedding_dim,
            embedding_size_lex=FLAGS.embedding_dim_lex,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda 
            )

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
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
        #print("Writing to {}\n".format(out_dir))

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


        """
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())
        """

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch, x_batch_lex=None):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              #lexicon
              cnn.input_x_lexicon: x_batch_lex,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            """
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print("{}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                 format(time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            """
            train_summary_writer.add_summary(summaries, step)

        def step(stepType,x_batch, y_batch, x_batch_lex=None, writer=None, lowerbound=57.5):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              #lexicon
              cnn.input_x_lexicon: x_batch_lex,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1, flattedStuff = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1, cnn.h_pool_flat],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: {}: step {}, loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(stepType, time_str, step, loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))
            if writer:
                writer.add_summary(summaries, step)

            if avg_f1>lowerbound:
                return True

            else:
                return False

        # Generate batches
        batches = cnn_data_helpers.batch_iter(
            list(zip(x_train, y_train, x_lex_train)), FLAGS.batch_size, FLAGS.num_epochs, FLAGS.random_seed)
        

        # Training loop. For each batch...

        for batch in batches:
            if FLAGS.do_not_train:
                break
            x_batch, y_batch, x_batch_lex = zip(*batch)
            train_step(x_batch, y_batch, x_batch_lex)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("Evaluation:")
                step("DEV", x_dev, y_dev, x_lex_dev, writer=dev_summary_writer)
                step("TEST", x_test, y_test, x_lex_test, writer=dev_summary_writer)
                print("")
