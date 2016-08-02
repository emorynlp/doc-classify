import cnn_data_helpers
import tensorflow as tf
from text_cnn import TextCNN

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 2.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("test_every", 614, "Evaluate model on test set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


max_len = 60
x_train, y_train, vocabulary, vocabulary_inv = cnn_data_helpers.load_data('trn', max_len)
x_test, y_test, _, _ = cnn_data_helpers.load_data('tst', max_len, vocabulary, vocabulary_inv)
# savepath = './models/model-8200'
# savepath = './models/model-13800'
# savepath = './models/model-3900'
# savepath = './models/model-5050'
savepath = './models/model-6750'

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=3,
            vocab_size=len(vocabulary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)


        saver = tf.train.Saver(tf.all_variables())
        saver.restore(sess,savepath)



        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1 = sess.run(
                [cnn.loss, cnn.accuracy,
                 cnn.neg_r, cnn.neg_p, cnn.f1_neg, cnn.f1_pos, cnn.avg_f1],
                feed_dict)

            print("loss {:g}, acc {:g}, neg_r {:g} neg_p {:g} f1_neg {:g}, f1_pos {:g}, f1 {:g}".
                  format(loss, accuracy, neg_r, neg_p, f1_neg, f1_pos, avg_f1))




        print("\nTest:")
        test_step(x_test, y_test)
        print("")

