import tensorflow as tf
import numpy as np


class TextCNNAttention(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.h_list=[]
        self.h_lex_list = []

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
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
                U_shape = [num_filters, num_filters_lex] # (256, 9)
                U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_E = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_E")
                b_E = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_E")

                conv_E = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W_E,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                w2v_conv = tf.nn.relu(tf.nn.bias_add(conv_E, b_E), name="relu_E") # (?, 59, 1, 256)
                self.h_list.append(w2v_conv)

                # for the lex
                filter_shape = [filter_size, embedding_size_lex, 1, num_filters_lex]
                W_L = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_L")
                b_L = tf.Variable(tf.constant(0.1, shape=[num_filters_lex]), name="b_L")

                conv_L = tf.nn.conv2d(
                    self.embedded_chars_expanded_lexicon,
                    W_L,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                lex_conv = tf.nn.relu(tf.nn.bias_add(conv_L, b_L), name="relu_L") # (?, 59, 1, 9)
                self.h_lex_list.append(lex_conv)

                w2v_sq = tf.squeeze(w2v_conv, squeeze_dims=[2]) # (?, 59, 256)
                lex_sq = tf.squeeze(lex_conv, squeeze_dims=[2]) # (?, 59, 9)

                print '[w2v_sq]', w2v_sq

                w2v_sq_tr = tf.batch_matrix_transpose(w2v_sq)
                print '[w2v_sq_tr]', w2v_sq_tr

                lex_sq_tr = tf.batch_matrix_transpose(lex_sq)
                print '[lex_sq_tr]', lex_sq_tr

                def fn(previous_output, current_input):
                    print(current_input.get_shape())
                    current_ouput = tf.matmul(U, current_input)
                    print 'previous_output', previous_output
                    print 'current_ouput', current_ouput
                    return current_ouput

                initializer = tf.constant(np.zeros([num_filters,59]), dtype=tf.float32)

                Ulex = tf.scan(fn, lex_sq_tr, initializer=initializer)
                print '[Ulex]', Ulex

                WUL = tf.batch_matmul(w2v_sq, Ulex)
                print '[WUL]', WUL


                WUL_expanded = tf.expand_dims(WUL, -1)
                print '[WUL_expanded]', WUL_expanded

                # Maxpooling over the outputs
                row_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, 1, sequence_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="row_pool")

                print '[row_pool]', row_pool

                col_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="col_pool")

                print '[col_pool]', col_pool

                row_pool_sq = tf.expand_dims(tf.squeeze(row_pool, squeeze_dims=[2, 3]), -1)  # (?, 59, 256)
                print '[row_pool_sq]', row_pool_sq

                col_pool_sq = tf.expand_dims(tf.squeeze(col_pool, squeeze_dims=[1, 3]), -1)  # (?, 59, 256)
                print '[col_pool_sq]', col_pool_sq

                print '[w2v_sq_tr]', w2v_sq_tr
                print '[lex_sq_tr]', lex_sq_tr

                attentioned_w2v = tf.batch_matmul(w2v_sq_tr, col_pool_sq)
                attentioned_lex = tf.batch_matmul(lex_sq_tr, row_pool_sq)

                attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
                attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

                print '[attentioned_w2v]', attentioned_w2v_sq
                print '[attentioned_lex]', attentioned_lex_sq


                pooled_outputs.append(attentioned_w2v_sq)
                pooled_outputs.append(attentioned_lex_sq)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        print '[pooled_outputs]', len(pooled_outputs)
        self.h_pool = tf.concat(1, pooled_outputs)
        print '[self.h_pool]', self.h_pool
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool_flat]', self.h_pool_flat

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)/30
            l2_loss += tf.nn.l2_loss(b)/30
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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


class TextCNNAttentionSimple(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.h_list=[]
        self.h_lex_list = []

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
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
                U_shape = [num_filters, num_filters_lex] # (256, 9)
                U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_E = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_E")
                b_E = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_E")

                conv_E = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W_E,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                w2v_conv = tf.nn.relu(tf.nn.bias_add(conv_E, b_E), name="relu_E") # (?, 59, 1, 256)
                self.h_list.append(w2v_conv)

                # for the lex
                filter_shape = [filter_size, embedding_size_lex, 1, num_filters_lex]
                W_L = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_L")
                b_L = tf.Variable(tf.constant(0.1, shape=[num_filters_lex]), name="b_L")

                conv_L = tf.nn.conv2d(
                    self.embedded_chars_expanded_lexicon,
                    W_L,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                lex_conv = tf.nn.relu(tf.nn.bias_add(conv_L, b_L), name="relu_L") # (?, 59, 1, 9)
                self.h_lex_list.append(lex_conv)

                w2v_sq = tf.squeeze(w2v_conv, squeeze_dims=[2]) # (?, 59, 256)
                lex_sq = tf.squeeze(lex_conv, squeeze_dims=[2]) # (?, 59, 9)

                print '[w2v_sq]', w2v_sq

                w2v_sq_tr = tf.batch_matrix_transpose(w2v_sq)
                print '[w2v_sq_tr]', w2v_sq_tr

                lex_sq_tr = tf.batch_matrix_transpose(lex_sq)
                print '[lex_sq_tr]', lex_sq_tr

                def fn(previous_output, current_input):
                    print(current_input.get_shape())
                    current_ouput = tf.matmul(U, current_input)
                    print 'previous_output', previous_output
                    print 'current_ouput', current_ouput
                    return current_ouput

                initializer = tf.constant(np.zeros([num_filters,59]), dtype=tf.float32)

                Ulex = tf.scan(fn, lex_sq_tr, initializer=initializer)
                print '[Ulex]', Ulex

                WUL = tf.batch_matmul(w2v_sq, Ulex)
                print '[WUL]', WUL


                WUL_expanded = tf.expand_dims(WUL, -1)
                print '[WUL_expanded]', WUL_expanded

                # Maxpooling over the outputs
                row_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, 1, sequence_length - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="row_pool")

                print '[row_pool]', row_pool

                col_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="col_pool")

                print '[col_pool]', col_pool

                row_pool_sq = tf.squeeze(row_pool, squeeze_dims=[2, 3])
                print '[row_pool_sq]', row_pool_sq

                col_pool_sq = tf.squeeze(col_pool, squeeze_dims=[1, 3])
                print '[col_pool_sq]', col_pool_sq

                # print '[w2v_sq_tr]', w2v_sq_tr
                # print '[lex_sq_tr]', lex_sq_tr

                pooled_outputs.append(row_pool_sq)
                pooled_outputs.append(col_pool_sq)


        # Combine all the pooled features
        num_filters_total = 59*2+58*2+57*2+56*2
        print '[pooled_outputs]', len(pooled_outputs)
        self.h_pool = tf.concat(1, pooled_outputs)
        print '[self.h_pool]', self.h_pool
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool_flat]', self.h_pool_flat

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)/30
            l2_loss += tf.nn.l2_loss(b)/30
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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



class TextCNNAttentionSimpleUT(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.h_list=[]
        self.h_lex_list = []

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
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
                U_shape = [sequence_length - filter_size + 1, sequence_length - filter_size + 1] # (59, 59)
                U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W_E = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_E")
                b_E = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_E")

                conv_E = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W_E,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                w2v_conv = tf.nn.relu(tf.nn.bias_add(conv_E, b_E), name="relu_E") # (?, 59, 1, 256)
                self.h_list.append(w2v_conv)

                # for the lex
                filter_shape = [filter_size, embedding_size_lex, 1, num_filters_lex]
                W_L = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_L")
                b_L = tf.Variable(tf.constant(0.1, shape=[num_filters_lex]), name="b_L")

                conv_L = tf.nn.conv2d(
                    self.embedded_chars_expanded_lexicon,
                    W_L,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                lex_conv = tf.nn.relu(tf.nn.bias_add(conv_L, b_L), name="relu_L") # (?, 59, 1, 9)
                self.h_lex_list.append(lex_conv)

                w2v_sq = tf.squeeze(w2v_conv, squeeze_dims=[2]) # (?, 59, 256)
                lex_sq = tf.squeeze(lex_conv, squeeze_dims=[2]) # (?, 59, 9)

                print '[w2v_sq]', w2v_sq

                w2v_sq_tr = tf.batch_matrix_transpose(w2v_sq)
                print '[w2v_sq_tr]', w2v_sq_tr

                lex_sq_tr = tf.batch_matrix_transpose(lex_sq)
                print '[lex_sq_tr]', lex_sq_tr

                def fn(previous_output, current_input):
                    print(current_input.get_shape())
                    current_ouput = tf.matmul(U, current_input)
                    print 'previous_output', previous_output
                    print 'current_ouput', current_ouput
                    return current_ouput

                initializer = tf.constant(np.zeros([num_filters,59]), dtype=tf.float32)

                Ulex = tf.scan(fn, lex_sq, initializer=initializer)
                print '[Ulex]', Ulex

                WUL = tf.batch_matmul(w2v_sq_tr, Ulex)
                print '[WUL]', WUL


                WUL_expanded = tf.expand_dims(WUL, -1)
                print '[WUL_expanded]', WUL_expanded

                # Maxpooling over the outputs
                row_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, 1, num_filters_lex, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="row_pool")

                print '[row_pool]', row_pool

                col_pool = tf.nn.max_pool(
                    WUL_expanded,
                    ksize=[1, num_filters, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="col_pool")

                print '[col_pool]', col_pool

                row_pool_sq = tf.squeeze(row_pool, squeeze_dims=[2, 3])
                print '[row_pool_sq]', row_pool_sq

                col_pool_sq = tf.squeeze(col_pool, squeeze_dims=[1, 3])
                print '[col_pool_sq]', col_pool_sq

                # print '[w2v_sq_tr]', w2v_sq_tr
                # print '[lex_sq_tr]', lex_sq_tr

                pooled_outputs.append(row_pool_sq)
                pooled_outputs.append(col_pool_sq)


        # Combine all the pooled features
        # num_filters_total = 59*2+58*2+57*2+56*2
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)

        print '[pooled_outputs]', len(pooled_outputs)
        self.h_pool = tf.concat(1, pooled_outputs)
        print '[self.h_pool]', self.h_pool
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool_flat]', self.h_pool_flat

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)/30
            l2_loss += tf.nn.l2_loss(b)/30
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
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

