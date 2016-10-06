import tensorflow as tf
import numpy as np


class W2V_LEX_CNN_MC(object):
    """
    A CNN for text classification. - multi channel version
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, l1_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size, 2], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l1_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars = self.input_x
            # self.embedded_chars_expanded = self.embedded_chars
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # print '[self.embedded_chars_expanded]', self.embedded_chars_expanded


        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

                conv = tf.nn.conv2d(
                    self.embedded_chars,
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


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

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
            l1_loss += tf.reduce_sum(tf.abs(W))
            l1_loss += tf.reduce_sum(tf.abs(b))
            self._b = b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss + l1_reg_lambda*l1_loss

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class W2V_LEX_CNN_MC_A2V(object):
    """
    A CNN for text classification. - multi channel version
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
            self, sequence_length, num_classes,
            embedding_size, filter_sizes, num_filters, embedding_size_lex,
            attention_depth_w2v, attention_depth_lex,
            l2_reg_lambda=0.0, l1_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x_2c = tf.placeholder(tf.float32, [None, sequence_length, embedding_size, 2], name="input_x_2c")
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        l1_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embedded_chars = self.input_x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print self.embedded_chars_expanded

            # lexicon embedding
            self.embedded_chars_lexicon = self.input_x_lexicon
            self.embedded_chars_expanded_lexicon = tf.expand_dims(self.embedded_chars_lexicon, -1)

            print '[self.embedded_chars]', self.embedded_chars
            print '[self.embedded_chars_expanded]', self.embedded_chars_expanded

            print '[self.embedded_chars_lexicon]', self.embedded_chars_lexicon
            print '[self.embedded_chars_expanded_lexicon]', self.embedded_chars_expanded_lexicon



        attention_outputs = []
        with tf.name_scope("pre-attention"):
            U_shape = [embedding_size, attention_depth_w2v]  # (400, 60)
            self.U_w2v = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_w2v")
            U_shape = [embedding_size_lex, attention_depth_lex]  # (15, 60)
            self.U_lex = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_lex")

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn_matmul_w2v(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_w2v)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            def fn_matmul_lex(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_lex)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            initializer = tf.constant(np.zeros([sequence_length, attention_depth_w2v]), dtype=tf.float32)
            WU_w2v = tf.scan(fn_matmul_w2v, self.embedded_chars, initializer=initializer)
            print '[WU_w2v]', WU_w2v

            initializer = tf.constant(np.zeros([sequence_length, attention_depth_lex]), dtype=tf.float32)
            LU_lex = tf.scan(fn_matmul_lex, self.embedded_chars_lexicon, initializer=initializer)
            print '[LU_lex]', LU_lex

            WU_w2v_expanded = tf.expand_dims(WU_w2v, -1)
            print '[WU_w2v_expanded]', WU_w2v_expanded  # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool  # (?, 60(seq_len), 1, 1) #select attention for w2v

            LU_lex_expanded = tf.expand_dims(LU_lex, -1)
            print '[LU_lex_expanded]', LU_lex_expanded  # (?, 60(seq_len), 60(depth), 1)

            lex_pool = tf.nn.max_pool(
                LU_lex_expanded,
                ksize=[1, 1, attention_depth_lex, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="lex_pool")

            print '[lex_pool]', lex_pool  # (?, 60(seq_len), 1, 1) #select attention for lex

            w2v_pool_sq = tf.expand_dims(tf.squeeze(w2v_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[w2v_pool_sq]', w2v_pool_sq

            lex_pool_sq = tf.expand_dims(tf.squeeze(lex_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[lex_pool_sq]', lex_pool_sq

            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, w2v_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, lex_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print '[attentioned_w2v]', attentioned_w2v_sq
            print '[attentioned_lex]', attentioned_lex_sq
            attention_outputs.append(attentioned_w2v_sq)
            attention_outputs.append(attentioned_lex_sq)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

                conv = tf.nn.conv2d(
                    self.input_x_2c,
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


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        self.appended_pool = tf.concat(1, [self.h_pool_flat, attention_outputs[0], attention_outputs[1]])
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = num_filters_total + embedding_size + embedding_size_lex

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.appended_pool, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)/30
            l2_loss += tf.nn.l2_loss(b)/30
            l1_loss += tf.reduce_sum(tf.abs(W))
            l1_loss += tf.reduce_sum(tf.abs(b))
            self._b = b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss + l1_reg_lambda*l1_loss

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2
