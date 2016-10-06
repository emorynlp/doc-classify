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
from cnn_models.w2v_trainable import W2V_TRAINABLE

import time
import gc
import re
import sys
import pickle

from tensorflow.contrib import learn


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
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

def run_train(w2vsource, w2vdim, w2vnumfilters, lexdim, lexnumfilters, randomseed, datasource, model_name, trainable, the_epoch):

    np.random.seed(randomseed)
    max_len = 60
    norm_model = []

    with Timer("lex"):
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
            fname = '../data/le/%s' % lexfile
            print 'default lexicon for %s' % lexfile

            with open(fname, 'rb') as handle:
                each_model = pickle.load(handle)
                default_vector = default_vector_dic[lexfile.replace('.pickle', '')]
                each_model["<PAD/>"] = default_vector
                norm_model.append(each_model)

    
    unigram_lexicon_model = norm_model


    # CONFIGURE
    # ==================================================
    if datasource == 'semeval':
        numberofclass = 3
        use_rotten_tomato = False
    elif datasource == 'sst':
        numberofclass = 5
        use_rotten_tomato = True


    # Training
    # ==================================================
    if randomseed > 0:
        tf.set_random_seed(randomseed)
    with tf.Graph().as_default():
        tf.set_random_seed(randomseed)
        max_af1_dev = 0
        index_at_max_af1_dev = 0
        af1_tst_at_max_af1_dev = 0

        #WORD2VEC
        x_text, y = cnn_data_helpers.load_data_trainable("everydata", rottenTomato=use_rotten_tomato)
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        vocab_processor.fit_transform(x_text)
        total_vocab_size = len(vocab_processor.vocabulary_)

        x_train, y_train = cnn_data_helpers.load_data_trainable("trn", rottenTomato=use_rotten_tomato)
        x_dev, y_dev = cnn_data_helpers.load_data_trainable("dev", rottenTomato=use_rotten_tomato)
        x_test, y_test = cnn_data_helpers.load_data_trainable("tst", rottenTomato=use_rotten_tomato)
        x_train = np.array(list(vocab_processor.fit_transform(x_train)))
        x_dev = np.array(list(vocab_processor.fit_transform(x_dev)))
        x_test = np.array(list(vocab_processor.fit_transform(x_test)))



        del(norm_model)
        gc.collect()

        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if randomseed > 0:
                tf.set_random_seed(randomseed)

            cnn = W2V_TRAINABLE(
                sequence_length=x_train.shape[1],
                num_classes=numberofclass,
                vocab_size=len(vocab_processor.vocabulary_),
                is_trainable=trainable,
                embedding_size=w2vdim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=w2vnumfilters,
                embedding_size_lex=lexdim,
                num_filters_lex=lexnumfilters,
                themodel=model_name,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )
           
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


            # initial matrix with random uniform
            initW = np.random.uniform(0.0,0.0,(total_vocab_size, w2vdim))
            initW_lex = np.random.uniform(0.00,0.2,(total_vocab_size, lexdim))
            # load any vectors from the word2vec
            with Timer("LOADING W2V..."):
                print("LOADING word2vec file {} \n".format(the_model_path))
                #W2V
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
            with Timer("LOADING LEXICON..."):
                vocabulary_set = set()
                for index, eachModel in enumerate(unigram_lexicon_model):
                    for word in eachModel:
                        vocabulary_set.add(word)

                for word in vocabulary_set:
                    lexiconList = np.empty([0, 1])
                    for index, eachModel in enumerate(unigram_lexicon_model):
                        if word in eachModel:
                            temp = np.array(np.float32(eachModel[word]))
                        else:
                            temp = np.array(np.float32(eachModel["<PAD/>"]))
                        lexiconList = np.append(lexiconList, temp)

                    idx = vocab_processor.vocabulary_.get(word)
                    if idx != 0:
                        initW_lex[idx] = lexiconList




            sess.run(cnn.W.assign(initW))
            if model_name == 'w2v_lex':
                sess.run(cnn.W_lex.assign(initW_lex))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
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
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None, score_type='f1'):
                """
                Evaluates model on a dev set
                """
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

            def test_step(x_batch, y_batch, writer=None, score_type='f1'):
                """
                Evaluates model on a test set
                """

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
            batches = cnn_data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, the_epoch)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)


                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    
                    print("Evaluation:")

                    if datasource == 'semeval':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer)

                    elif datasource == 'sst':
                        curr_af1_dev = dev_step(x_dev, y_dev, writer=dev_summary_writer, score_type = 'acc')
                        curr_af1_tst = test_step(x_test, y_test, writer=test_summary_writer, score_type = 'acc')


                    if curr_af1_dev > max_af1_dev:
                        max_af1_dev = curr_af1_dev
                        index_at_max_af1_dev = current_step
                        af1_tst_at_max_af1_dev = curr_af1_tst

                    print 'Status: [%d] Max f1 for dev (%f), Max f1 for tst (%f)\n' % (
                        index_at_max_af1_dev, max_af1_dev, af1_tst_at_max_af1_dev)
                    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--w2vsource', default='twitter', choices=['twitter','amazon'], type=str)
    parser.add_argument('--w2vdim', default=50, type=int)
    parser.add_argument('--w2vnumfilters', default=16, type=int)
    parser.add_argument('--lexdim', default=15, type=int)
    parser.add_argument('--lexnumfilters', default=9, type=int)
    parser.add_argument('--randomseed', default=7, type=int)
    parser.add_argument('--datasource', default='semeval', choices=['semeval','sst'], type=str)
    parser.add_argument('--model', default='w2v_lex', choices=['w2v', 'w2v_lex'], type=str) # w2v, w2vlex, attention
    parser.add_argument('--trainable', default='nonstatic', choices=['static', 'nonstatic'], type=str) # w2v, w2vlex, attention
    parser.add_argument('--epoches', default=30, type=int)

    args = parser.parse_args()
    program = os.path.basename(sys.argv[0])

    print 'ADDITIONAL PARAMETER\n w2vsource: %s\n w2vdim: %d\n w2vnumfilters: %d\n lexdim: %d\n lexnumfilters: %d\n ' \
          'randomseed: %d\n data_source: %s\n model_name: %s\n trainable: %s\n epoch: %d' \
          % (args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed, args.datasource,
             args.model, args.trainable, args.epoches)

    run_train(args.w2vsource, args.w2vdim, args.w2vnumfilters, args.lexdim, args.lexnumfilters, args.randomseed,
              args.datasource, args.model, args.trainable, args.epoches)
