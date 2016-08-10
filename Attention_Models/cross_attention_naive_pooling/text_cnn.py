import numpy as np
import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes,
      embedding_size, filter_sizes, num_filters, embedding_size_lex=2, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        #lexicon input
        self.input_x_lexicon = tf.placeholder(tf.float32, [None, sequence_length, embedding_size_lex], name="input_x_lexicon")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            #w2v embedding
            self.embedded_chars = self.input_x
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print('embedded_chars: '),
            print(self.embedded_chars)
            print('embedded_chars_expanded: '),
            print(self.embedded_chars_expanded)
            #lexicon embedding
            self.embedded_chars_lexicon = self.input_x_lexicon
            self.embedded_chars_expanded_lexicon = tf.expand_dims(self.embedded_chars_lexicon, -1)
            print('embedded_chars_lex: '),
            print(self.embedded_chars_lexicon)
            print('embedded_chars_expanded_lex: '),
            print(self.embedded_chars_expanded_lexicon)
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # Both lexicon and w2v so cross attention can be used
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer w2v
                filter_shape_w2v = [filter_size, embedding_size, 1, num_filters]

                W_w2v = tf.Variable(tf.truncated_normal(filter_shape_w2v, stddev=0.1), name="W_w2v")
                b_w2v = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_w2v")

                conv_w2v = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W_w2v,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv_w2v")

                conv_w2v = tf.nn.bias_add(conv_w2v, b_w2v)

		# Convolution Layer lex
		filter_shape_lex = [filter_size, embedding_size_lex, 1, num_filters]
		W_lex = tf.Variable(tf.truncated_normal(filter_shape_lex, stddev=0.1), name="W_lex")
		b_lex = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_lex")

		conv_lex = tf.nn.conv2d(
			self.embedded_chars_expanded_lexicon,
			W_lex,
			strides=[1, 1, 1, 1],
			padding="VALID",
			name="conv_lex")

		conv_lex = tf.nn.bias_add(conv_lex, b_lex)


		# Reshape for multiplication along num_filters dimension
		conv_w2v_4matmul = tf.reshape(conv_w2v, [-1, num_filters], name="conv_w2v_4matmul")
                conv_lex_4matmul = tf.reshape(conv_lex, [-1, num_filters], name="conv_lex_4matmul")
                transp_conv_lex_4matmul = tf.transpose(conv_lex_4matmul)


                # Multiply crossed by attention and tanh to norm
                cross = tf.matmul(transp_conv_lex_4matmul, conv_w2v_4matmul)
                attn = tf.Variable(tf.truncated_normal([num_filters, num_filters], stddev=0.1), name="U")
                cross_attn = tf.matmul(cross, attn)
	        cross_attn_norm = tf.tanh(cross_attn, name="cross_attn_norm")

	        # Multiply original convolutional outputs by the learned attention matrix
                w2v_attn = tf.matmul(conv_w2v_4matmul, cross_attn_norm)
                lex_attn_tsp= tf.matmul(cross_attn_norm, transp_conv_lex_4matmul)
                lex_attn = tf.transpose(lex_attn_tsp)

                # Reshape output back to form
                w2v_attn_4pool = tf.reshape(w2v_attn, [-1, (sequence_length - filter_size + 1), 1, num_filters])
                lex_attn_4pool = tf.reshape(lex_attn, [-1, (sequence_length - filter_size + 1), 1, num_filters])

                # Normalization
                h_w2v = tf.nn.relu(w2v_attn_4pool, name="relu_w2v")
                h_lex = tf.nn.relu(lex_attn_4pool, name='relu_lex')

                # Maxpooling over the outputs -- only doing regular,
                # need to do row-wise and column-wise
                pooled_w2v = tf.nn.max_pool(
                    h_w2v,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_w2v")

                pooled_outputs.append(pooled_w2v)

                pooled_lex = tf.nn.max_pool(
                    h_lex,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool_lex")

                pooled_outputs.append(pooled_lex)



        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) * 2
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print ("PRINTING POOL FLAT")
        print (self.h_pool_flat)
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

