import tensorflow as tf
import numpy as np


class MultiClassRegression(object):
    """
    A logistic regression for cancer classification.
    """
    def __init__(self, num_features, num_output, l2_reg_lambda=0.0, neg_output=False):
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_output], name="input_y")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("softmax"):
            filter_shape = [num_features, num_output]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_output]))

            self.raw_scores = tf.nn.xw_plus_b(self.input_x, W, b, name="scores")
            if neg_output:
                self.scores = tf.nn.elu(self.raw_scores, name="tanh")

            else:
                self.scores = tf.nn.relu(self.raw_scores, name="relu")


            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("loss"):
            self.losses = tf.square(tf.sub(self.scores, self.input_y))
            self.avgloss = tf.reduce_mean(tf.abs(tf.sub(self.scores, self.input_y)))
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss


class MultiClassRegressionPrediction(object):
    """
    A logistic regression for cancer classification.
    """
    def __init__(self, num_features, num_output, neg_output=False):
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name="input_x")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("softmax"):
            filter_shape = [num_features, num_output]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_output]))

            self.raw_scores = tf.nn.xw_plus_b(self.input_x, W, b, name="scores")
            if neg_output:
                self.scores = tf.nn.elu(self.raw_scores, name="tanh")

            else:
                self.scores = tf.nn.relu(self.raw_scores, name="relu")




class MultiClassRegression2Layer(object):
    """
    A logistic regression for cancer classification.
    """
    def __init__(self, num_features, num_output, l2_reg_lambda=0.0):
        self.input_x = tf.placeholder(tf.float32, [None, num_features], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_output], name="input_y")
        num_hidden1 = 500

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.name_scope("layer-1"):
            filter_shape = [num_features, num_hidden1]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_hidden1]))

            self.h1_raw = tf.nn.xw_plus_b(self.input_x, W, b, name="h1_raw")
            self.h1 = tf.nn.relu(self.h1_raw, name="h1_relu")
            # self.h1 = tf.nn.tanh(self.h1_raw, name="h1_sigmoid")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("layer-2"):
            filter_shape = [num_hidden1, num_output]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_output]))

            self.raw_scores = tf.nn.xw_plus_b(self.h1, W, b, name="raw_scores")
            self.scores = tf.nn.elu(self.raw_scores, name="relu_scores")
            # self.scores = tf.nn.sigmoid(self.raw_scores, name="relu")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

        with tf.name_scope("loss"):
            self.losses = tf.square(tf.sub(self.scores, self.input_y))
            self.avgloss = tf.reduce_mean(tf.abs(tf.sub(self.scores, self.input_y)))
            self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda * l2_loss