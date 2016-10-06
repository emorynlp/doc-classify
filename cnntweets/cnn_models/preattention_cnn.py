import tensorflow as tf
import numpy as np

# Plain attention before cnn
# most attention's output will be zero because of coverage of lexicon is small
# so expanded lexicon is required, and this helps
class TextCNNPreAttention(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0,
            l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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
            U_shape = [embedding_size, embedding_size_lex]  # (400, 15)
            self.U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")

            # self.UL = self.embedded_chars_lexicon
            # self.embedded_chars # (?, 60, 400)
            # self.embedded_chars_lexicon # (?, 60, 15)

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(self.U, current_input)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            initializer = tf.constant(np.zeros([embedding_size, sequence_length]), dtype=tf.float32)
            Ulex = tf.scan(fn, self.embedded_chars_lexicon_tr, initializer=initializer)
            print '[Ulex]', Ulex

            self.WUL = tf.batch_matmul(self.embedded_chars, Ulex)
            print '[self.WUL]', self.WUL # (?, 60, 60) 60(w2v) x 60(lex)

            WUL_expanded = tf.expand_dims(self.WUL, -1)
            print '[WUL_expanded]', WUL_expanded # (?, 60, 60, 1)

            row_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, 1, sequence_length, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="row_pool")

            print '[row_pool]', row_pool # (?, 60, 1, 1) #select attention for w2v

            col_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, sequence_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="col_pool")

            print '[col_pool]', col_pool  # (?, 1, 60, 1) #select attention for lex

            row_pool_sq = tf.expand_dims(tf.squeeze(row_pool, squeeze_dims=[2, 3]), -1)  # (?, 59, 256)
            print '[row_pool_sq]', row_pool_sq

            col_pool_sq = tf.expand_dims(tf.squeeze(col_pool, squeeze_dims=[1, 3]), -1)  # (?, 59, 256)
            print '[col_pool_sq]', col_pool_sq


            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, row_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, col_pool_sq)
            # attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, col_pool_sq)
            # attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, row_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print '[attentioned_w2v]', attentioned_w2v_sq
            print '[attentioned_lex]', attentioned_lex_sq
            attention_outputs.append(attentioned_w2v_sq)
            attention_outputs.append(attentioned_lex_sq)

        # Create a convolution + maxpool layer for each filter size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

# Plain attention before cnn plus bias (since most attention's output will be zero -> add bias to it)
class TextCNNPreAttentionBias(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0,
            l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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
            U_shape = [embedding_size, embedding_size_lex]  # (400, 15)
            self.U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")
            Ub_shape = [sequence_length, sequence_length] # (60, 60)
            self.Ub = tf.Variable(tf.truncated_normal(Ub_shape, stddev=0.1), name="Ub")

            # self.UL = self.embedded_chars_lexicon
            # self.embedded_chars # (?, 60, 400)
            # self.embedded_chars_lexicon # (?, 60, 15)

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn_matmul(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(self.U, current_input)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            initializer = tf.constant(np.zeros([embedding_size, sequence_length]), dtype=tf.float32)
            Ulex = tf.scan(fn_matmul, self.embedded_chars_lexicon_tr, initializer=initializer)
            print '[Ulex]', Ulex

            self.WUL = tf.batch_matmul(self.embedded_chars, Ulex)
            print '[self.WUL]', self.WUL # (?, 60, 60) 60(w2v) x 60(lex)

            def fn_matadd(previous_output, current_input):
                print(current_input.get_shape())
                current_output = current_input + self.Ub
                print 'previous_output', previous_output
                print 'current_ouput', current_output
                return current_output

            initializer_Ub = tf.constant(np.zeros([sequence_length, sequence_length]), dtype=tf.float32)
            Ulex = tf.scan(fn_matadd, self.WUL, initializer=initializer_Ub)

            WUL_expanded = tf.expand_dims(self.WUL, -1)
            print '[WUL_expanded]', WUL_expanded # (?, 60, 60, 1)

            row_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, 1, sequence_length, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="row_pool")

            print '[row_pool]', row_pool # (?, 60, 1, 1) #select attention for w2v

            col_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, sequence_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="col_pool")

            print '[col_pool]', col_pool  # (?, 1, 60, 1) #select attention for lex

            row_pool_sq = tf.expand_dims(tf.squeeze(row_pool, squeeze_dims=[2, 3]), -1)  # (?, 59, 256)
            print '[row_pool_sq]', row_pool_sq

            col_pool_sq = tf.expand_dims(tf.squeeze(col_pool, squeeze_dims=[1, 3]), -1)  # (?, 59, 256)
            print '[col_pool_sq]', col_pool_sq


            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, row_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, col_pool_sq)
            # attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, col_pool_sq)
            # attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, row_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print '[attentioned_w2v]', attentioned_w2v_sq
            print '[attentioned_lex]', attentioned_lex_sq
            attention_outputs.append(attentioned_w2v_sq)
            attention_outputs.append(attentioned_lex_sq)

        # Create a convolution + maxpool layer for each filter size

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextAttention2Vec(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, l2_reg_lambda=0.0,
            l1_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
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
            U_shape = [embedding_size, embedding_size_lex]  # (400, 15)
            self.U = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U")

            # self.UL = self.embedded_chars_lexicon
            # self.embedded_chars # (?, 60, 400)
            # self.embedded_chars_lexicon # (?, 60, 15)

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(self.U, current_input)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            initializer = tf.constant(np.zeros([embedding_size, sequence_length]), dtype=tf.float32)
            Ulex = tf.scan(fn, self.embedded_chars_lexicon_tr, initializer=initializer)
            print '[Ulex]', Ulex

            self.WUL = tf.batch_matmul(self.embedded_chars, Ulex)
            print '[self.WUL]', self.WUL # (?, 60, 60) 60(w2v) x 60(lex)

            WUL_expanded = tf.expand_dims(self.WUL, -1)
            print '[WUL_expanded]', WUL_expanded # (?, 60, 60, 1)

            row_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, 1, sequence_length, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="row_pool")

            print '[row_pool]', row_pool # (?, 60, 1, 1) #select attention for w2v

            col_pool = tf.nn.max_pool(
                WUL_expanded,
                ksize=[1, sequence_length, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="col_pool")

            print '[col_pool]', col_pool  # (?, 1, 60, 1) #select attention for lex

            row_pool_sq = tf.expand_dims(tf.squeeze(row_pool, squeeze_dims=[2, 3]), -1)  # (?, 59, 256)
            print '[row_pool_sq]', row_pool_sq

            col_pool_sq = tf.expand_dims(tf.squeeze(col_pool, squeeze_dims=[1, 3]), -1)  # (?, 59, 256)
            print '[col_pool_sq]', col_pool_sq


            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, row_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, col_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print '[attentioned_w2v]', attentioned_w2v_sq
            print '[attentioned_lex]', attentioned_lex_sq
            attention_outputs.append(attentioned_w2v_sq)
            attention_outputs.append(attentioned_lex_sq)



        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = tf.concat(1, [attention_outputs[0], attention_outputs[1]])
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = embedding_size + embedding_size_lex

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextAttention2VecIndividual(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
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
            print '[WU_w2v_expanded]', WU_w2v_expanded # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool # (?, 60(seq_len), 1, 1) #select attention for w2v

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



        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = tf.concat(1, [attention_outputs[0], attention_outputs[1]])
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = embedding_size + embedding_size_lex

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextAttention2VecIndividualBias(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
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
            U_shape_w2v = [embedding_size, attention_depth_w2v]  # (400, 60)
            self.U_w2v = tf.Variable(tf.truncated_normal(U_shape_w2v, stddev=0.1), name="U_w2v")
            U_shape_w2v_b = [sequence_length, attention_depth_w2v]  # (60, 20)
            self.U_w2v_b = tf.Variable(tf.truncated_normal(U_shape_w2v_b, stddev=0.1), name="U_w2v_b")

            U_shape_lex = [embedding_size_lex, attention_depth_lex]  # (15, 60)
            self.U_lex = tf.Variable(tf.truncated_normal(U_shape_lex, stddev=0.1), name="U_lex")
            U_shape_lex_b = [sequence_length, attention_depth_lex]  # (60, 20)
            self.U_lex_b = tf.Variable(tf.truncated_normal(U_shape_lex_b, stddev=0.1), name="U_lex_b")

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn_matmul_w2v(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_w2v) + self.U_w2v_b
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            def fn_matmul_lex(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_lex) + self.U_lex_b
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
            print '[WU_w2v_expanded]', WU_w2v_expanded # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool # (?, 60(seq_len), 1, 1) #select attention for w2v

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



        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = tf.concat(1, [attention_outputs[0], attention_outputs[1]])
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = embedding_size + embedding_size_lex

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextAttention2VecIndividualW2v(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
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
            print '[WU_w2v_expanded]', WU_w2v_expanded # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool # (?, 60(seq_len), 1, 1) #select attention for w2v

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



        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

        # self.appended_pool = tf.concat(1, [attention_outputs[0], attention_outputs[1]])
        self.appended_pool = attention_outputs[0]
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = embedding_size

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextAttention2VecIndividualLex(object):
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
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
            print '[WU_w2v_expanded]', WU_w2v_expanded # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool # (?, 60(seq_len), 1, 1) #select attention for w2v

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



        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = attention_outputs[1]
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = embedding_size_lex

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextCNNAttention2VecIndividual(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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

            self.w2v_pool_sq = tf.expand_dims(tf.squeeze(w2v_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[w2v_pool_sq]', self.w2v_pool_sq

            self.lex_pool_sq = tf.expand_dims(tf.squeeze(lex_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[lex_pool_sq]', self.lex_pool_sq

            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, self.w2v_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, self.lex_pool_sq)

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
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

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
            self._b = b
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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextCNNAttention2VecIndividualL1(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
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
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        print 'attention_outputs[1]', attention_outputs[1]

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

class TextCNNAttention2VecIndividualW2v(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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
            # U_shape = [embedding_size_lex, attention_depth_lex]  # (15, 60)
            # self.U_lex = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_lex")

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            def fn_matmul_w2v(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_w2v)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            # def fn_matmul_lex(previous_output, current_input):
            #     print(current_input.get_shape())
            #     current_ouput = tf.matmul(current_input, self.U_lex)
            #     print 'previous_output', previous_output
            #     print 'current_ouput', current_ouput
            #     return current_ouput

            initializer = tf.constant(np.zeros([sequence_length, attention_depth_w2v]), dtype=tf.float32)
            WU_w2v = tf.scan(fn_matmul_w2v, self.embedded_chars, initializer=initializer)
            print '[WU_w2v]', WU_w2v

            # initializer = tf.constant(np.zeros([sequence_length, attention_depth_lex]), dtype=tf.float32)
            # LU_lex = tf.scan(fn_matmul_lex, self.embedded_chars_lexicon, initializer=initializer)
            # print '[LU_lex]', LU_lex

            WU_w2v_expanded = tf.expand_dims(WU_w2v, -1)
            print '[WU_w2v_expanded]', WU_w2v_expanded  # (?, 60(seq_len), 60(depth), 1)

            w2v_pool = tf.nn.max_pool(
                WU_w2v_expanded,
                ksize=[1, 1, attention_depth_w2v, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="w2v_pool")

            print '[w2v_pool]', w2v_pool  # (?, 60(seq_len), 1, 1) #select attention for w2v

            # LU_lex_expanded = tf.expand_dims(LU_lex, -1)
            # print '[LU_lex_expanded]', LU_lex_expanded  # (?, 60(seq_len), 60(depth), 1)

            # lex_pool = tf.nn.max_pool(
            #     LU_lex_expanded,
            #     ksize=[1, 1, attention_depth_lex, 1],
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name="lex_pool")

            # print '[lex_pool]', lex_pool  # (?, 60(seq_len), 1, 1) #select attention for lex

            w2v_pool_sq = tf.expand_dims(tf.squeeze(w2v_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[w2v_pool_sq]', w2v_pool_sq

            # lex_pool_sq = tf.expand_dims(tf.squeeze(lex_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            # print '[lex_pool_sq]', lex_pool_sq

            attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, w2v_pool_sq)
            # attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, lex_pool_sq)

            attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            # attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            print '[attentioned_w2v]', attentioned_w2v_sq
            # print '[attentioned_lex]', attentioned_lex_sq
            attention_outputs.append(attentioned_w2v_sq)
            # attention_outputs.append(attentioned_lex_sq)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        # print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = tf.concat(1, [self.h_pool_flat, attention_outputs[0]])
        print '[self.appended_pool]', self.appended_pool
        # num_filters_total = num_filters_total + embedding_size + embedding_size_lex
        num_filters_total = num_filters_total + embedding_size

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

class TextCNNAttention2VecIndividualLex(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex, num_filters_lex, attention_depth_w2v,
            attention_depth_lex, l2_reg_lambda=0.0, l1_reg_lambda=0.0):
        # sequence_length_lex = 30
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex],
                                              name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

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
            # U_shape = [embedding_size, attention_depth_w2v]  # (400, 60)
            # self.U_w2v = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_w2v")
            U_shape = [embedding_size_lex, attention_depth_lex]  # (15, 60)
            self.U_lex = tf.Variable(tf.truncated_normal(U_shape, stddev=0.1), name="U_lex")

            self.embedded_chars_tr = tf.batch_matrix_transpose(self.embedded_chars)
            self.embedded_chars_lexicon_tr = tf.batch_matrix_transpose(self.embedded_chars_lexicon)
            print '[self.embedded_chars_lexicon_tr]', self.embedded_chars_lexicon_tr

            # def fn_matmul_w2v(previous_output, current_input):
            #     print(current_input.get_shape())
            #     current_ouput = tf.matmul(current_input, self.U_w2v)
            #     print 'previous_output', previous_output
            #     print 'current_ouput', current_ouput
            #     return current_ouput

            def fn_matmul_lex(previous_output, current_input):
                print(current_input.get_shape())
                current_ouput = tf.matmul(current_input, self.U_lex)
                print 'previous_output', previous_output
                print 'current_ouput', current_ouput
                return current_ouput

            # initializer = tf.constant(np.zeros([sequence_length, attention_depth_w2v]), dtype=tf.float32)
            # WU_w2v = tf.scan(fn_matmul_w2v, self.embedded_chars, initializer=initializer)
            # print '[WU_w2v]', WU_w2v

            initializer = tf.constant(np.zeros([sequence_length, attention_depth_lex]), dtype=tf.float32)
            LU_lex = tf.scan(fn_matmul_lex, self.embedded_chars_lexicon, initializer=initializer)
            print '[LU_lex]', LU_lex

            # WU_w2v_expanded = tf.expand_dims(WU_w2v, -1)
            # print '[WU_w2v_expanded]', WU_w2v_expanded  # (?, 60(seq_len), 60(depth), 1)

            # w2v_pool = tf.nn.max_pool(
            #     WU_w2v_expanded,
            #     ksize=[1, 1, attention_depth_w2v, 1],
            #     strides=[1, 1, 1, 1],
            #     padding='VALID',
            #     name="w2v_pool")
            #
            # print '[w2v_pool]', w2v_pool  # (?, 60(seq_len), 1, 1) #select attention for w2v

            LU_lex_expanded = tf.expand_dims(LU_lex, -1)
            print '[LU_lex_expanded]', LU_lex_expanded  # (?, 60(seq_len), 60(depth), 1)

            lex_pool = tf.nn.max_pool(
                LU_lex_expanded,
                ksize=[1, 1, attention_depth_lex, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="lex_pool")

            print '[lex_pool]', lex_pool  # (?, 60(seq_len), 1, 1) #select attention for lex

            # w2v_pool_sq = tf.expand_dims(tf.squeeze(w2v_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            # print '[w2v_pool_sq]', w2v_pool_sq

            lex_pool_sq = tf.expand_dims(tf.squeeze(lex_pool, squeeze_dims=[2, 3]), -1)  # (?, 60, 1)
            print '[lex_pool_sq]', lex_pool_sq

            # attentioned_w2v = tf.batch_matmul(self.embedded_chars_tr, w2v_pool_sq)
            attentioned_lex = tf.batch_matmul(self.embedded_chars_lexicon_tr, lex_pool_sq)

            # attentioned_w2v_sq = tf.squeeze(attentioned_w2v, squeeze_dims=[2])
            attentioned_lex_sq = tf.squeeze(attentioned_lex, squeeze_dims=[2])

            # print '[attentioned_w2v]', attentioned_w2v_sq
            print '[attentioned_lex]', attentioned_lex_sq
            # attention_outputs.append(attentioned_w2v_sq)
            attention_outputs.append(attentioned_lex_sq)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # l2_loss += tf.nn.l2_loss(W)/1000
                # l2_loss += tf.nn.l2_loss(b)/1000

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

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) + num_filters_lex * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print '[self.h_pool]', self.h_pool
        print '[self.h_pool_flat]', self.h_pool_flat

        print 'pooled_outputs[0]', pooled_outputs[0]
        print 'pooled_outputs[1]', pooled_outputs[1]

        print 'attention_outputs[0]', attention_outputs[0]
        # print 'attention_outputs[1]', attention_outputs[1]

        self.appended_pool = tf.concat(1, [self.h_pool_flat, attention_outputs[0]])
        print '[self.appended_pool]', self.appended_pool
        num_filters_total = num_filters_total + embedding_size_lex

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

            self.f1_neg = 2 * self.neg_p * self.neg_r / (self.neg_p + self.neg_r) * 100
            self.f1_pos = 2 * pos_p * pos_r / (pos_p + pos_r) * 100

            self.avg_f1 = (self.f1_neg + self.f1_pos) / 2

