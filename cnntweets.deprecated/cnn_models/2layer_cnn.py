import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, lex_filter_size, permsel, l2_reg_lambda=0.0):
        num_filters_lex = lex_filter_size
        # num_filters_lex = 256

        if permsel==0:
            perm = [0, 1, 2, 3, 4, 5, 6, 7]

        elif permsel==1:
            perm = [3, 2, 1, 0, 4, 5, 6, 7]

        elif permsel==2:
            perm = [0, 1, 2, 3, 7, 6, 5, 4]

        else:
            perm = [0, 4, 1, 5, 2, 6, 3, 7]

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        self.input_for_hidden = tf.placeholder(tf.float32, [None, 8, 256], name="input_for_hidden")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #     name="W")
            # self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars = self.input_x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print self.embedded_chars_expanded

            # lexicon embedding
            self.embedded_chars_lexicon = self.input_x_lexicon
            self.embedded_chars_expanded_lexicon = tf.expand_dims(self.embedded_chars_lexicon, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # APPLY CNN TO LEXICON EMBEDDING
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("lexicon-conv-maxpool-%s" % filter_size):
                # Convolution Layer

                filter_shape = [filter_size, embedding_size_lex, 1, num_filters_lex]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters_lex]), name="b")

                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded_lexicon,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        x = tf.Variable(0)
        self.len_outputs = x.assign(len(pooled_outputs))
        self.pool_shape = tf.shape(pooled_outputs[0])


        self.pooled_outputs = pooled_outputs
        # construct a matrix
        hidden_sequence_length = len(pooled_outputs)
        hidden_embedding_size = 256
        input_hidden = np.zeros((hidden_sequence_length, hidden_embedding_size))


        pooled_outputs_perm = [pooled_outputs[i] for i in perm]

        # 8 kinds of filters, 4 from w2v, 4 from lex
        # for i in range(hidden_sequence_length):
        #     # input_hidden[i, :] = pooled_outputs[perm[i]][1, 1, 1, :]
        #     h_slice = tf.slice(pooled_outputs[perm[i]], [0, 0, 0, 0], [1, 1, 1, 256])
        #
        #     input_hidden[i, :] = h_slice

        num_filters_total = num_filters * len(filter_sizes) * 2

        self.h_pool = tf.concat(3, pooled_outputs_perm)
        self.h_pool_shape = tf.reshape(self.h_pool, [-1, 8, 256])
        self.h_pool_shape_expanded = tf.expand_dims(self.h_pool_shape, -1)

        scn_pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("2nd-conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, 256, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                conv = tf.nn.conv2d(
                    self.h_pool_shape_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, hidden_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                scn_pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # num_filters_total = num_filters * len(filter_sizes) * 2
        self.h_pool2 = tf.concat(3, scn_pooled_outputs)
        self.h_pool_flat2 = tf.reshape(self.h_pool2, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop2 = tf.nn.dropout(self.h_pool_flat2, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)/30
            l2_loss += tf.nn.l2_loss(b)/30
            self.scores = tf.nn.xw_plus_b(self.h_drop2, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.golds = tf.argmax(self.input_y, 1, name="golds")
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


        with tf.name_scope("avg_f1"):
            self.golds = tf.argmax(self.input_y, 1, name="golds")
            self.preds = self.predictions

            # positive recall
            pos_gold_sel = tf.equal(self.golds, 2)  # positive_gold
            posg_golds = tf.boolean_mask(self.golds, pos_gold_sel)
            posg_preds = tf.boolean_mask(self.preds, pos_gold_sel)
            correct_predictions_pr = tf.equal(posg_golds, posg_preds)
            pos_r = tf.reduce_mean(tf.cast(correct_predictions_pr, "float"), name="pos_recall")

            # positive precision
            pos_pred_sel = tf.equal(self.preds, 2)  # positive_pred
            posp_golds = tf.boolean_mask(self.golds, pos_pred_sel)
            posp_preds = tf.boolean_mask(self.preds, pos_pred_sel)
            correct_predictions_pp = tf.equal(posp_golds, posp_preds)
            pos_p = tf.reduce_mean(tf.cast(correct_predictions_pp, "float"), name="pos_precision")

            # negative recall
            neg_gold_sel = tf.equal(self.golds, 0)  # positive_gold
            negg_golds = tf.boolean_mask(self.golds, neg_gold_sel)
            negg_preds = tf.boolean_mask(self.preds, neg_gold_sel)
            correct_predictions_nr = tf.equal(negg_golds, negg_preds)
            self.neg_r = tf.reduce_mean(tf.cast(correct_predictions_nr, "float"), name="neg_recall")

            # negative precision
            neg_pred_sel = tf.equal(self.preds, 0)  # positive_pred
            negp_golds = tf.boolean_mask(self.golds, neg_pred_sel)
            negp_preds = tf.boolean_mask(self.preds, neg_pred_sel)
            correct_predictions_np = tf.equal(negp_golds, negp_preds)
            self.neg_p = tf.reduce_mean(tf.cast(correct_predictions_np, "float"), name="neg_precision")

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r + 0.00001) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r + 0.00001) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2


    def f1_score(pred_y, gold_y):
        prec_neg_cnt = []
        prec_pos_cnt = []
        recall_neg_cnt = []
        recall_pos_cnt = []
        acc = []

        Y=gold_y
        for i in range(len(gold_y)):
            if gold_y[i] == pred_y[i]:
                acc.append(1)
            else:
                acc.append(0)

            index = pred_y[i]
            # if pred is neg
            if index==2:
                if Y[i]==2:
                    prec_neg_cnt.append(1)
                else:
                    prec_neg_cnt.append(0)

            # if pred is positive
            if index==0:
                if Y[i]==0:
                    prec_pos_cnt.append(1)
                else:
                    prec_pos_cnt.append(0)

            # if gold is negative
            if Y[i]==2:
                if index==2:  # (pos, neg, obj)
                    recall_neg_cnt.append(1)
                else:
                    recall_neg_cnt.append(0)

            # if gold is positive (neg obj pos)
            if Y[i]==0:
                if index==0:# (pos, neg, obj)
                    recall_pos_cnt.append(1)
                else:
                    recall_pos_cnt.append(0)

        pr_neg = sum(prec_neg_cnt)*1.0/len(prec_neg_cnt)
        pr_pos = sum(prec_pos_cnt)*1.0/len(prec_pos_cnt)
        rc_neg = sum(recall_neg_cnt)*1.0/len(recall_neg_cnt)
        rc_pos = sum(recall_pos_cnt)*1.0/len(recall_pos_cnt)

        f1_neg = 2*pr_neg*rc_neg/(pr_neg+rc_neg)*100
        f1_pos = 2*pr_pos*rc_pos/(pr_pos+rc_pos)*100

        f1 =  (f1_neg+f1_pos)/2

        accuracy = sum(acc)*1.0/len(acc)*100

        rval=[]
        rval.append(f1_neg)
        rval.append(f1_pos)
        rval.append(f1)
        rval.append(accuracy)


        return rval

