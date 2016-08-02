# import pickle
# import numpy
#
# from tensorflow.contrib.learn.python.learn.datasets import base
# from tensorflow.python.framework import dtypes
# import numpy as np
# import re
# import itertools
# from collections import Counter
#
# BASE_URL = 'http://www.mathcs.emory.edu/bshin/'
#
# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()
#
#
# def load_data_and_labels(dataset):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     template_txt = '../data/tweets/txt/%s'
#
#
#     pathtxt = template_txt % dataset
#     # pathtxt = template_txt % 'tst'
#     # pathtxt = template_txt % 'trn'
#     # pathtxt = template_txt % 'dev'
#
#     x_text=[line.split('\t')[2] for line in open(pathtxt, "r").readlines()]
#     x_text = [clean_str(sent) for sent in x_text]
#     x_text = [s.split(" ") for s in x_text]
#
#     y = []
#     for line in open(pathtxt, "r").readlines():
#         senti=line.split('\t')[1]
#         if  senti == 'objective':
#             y.append([0, 1, 0])
#
#         elif senti == 'positive':
#             y.append([0, 0, 1])
#
#         else:  # negative
#             y.append([1, 0, 0])
#
#     return [x_text, y]
#
#
# def pad_sentences(sentences, padlen, padding_word="<PAD/>"):
#     """
#     Pads all sentences to the same length. The length is defined by the longest sentence.
#     Returns padded sentences.
#     """
#     if padlen==None:
#         sequence_length = max(len(x) for x in sentences)
#     else:
#         sequence_length=padlen
#
#     padded_sentences = []
#     for i in range(len(sentences)):
#         sentence = sentences[i]
#         num_padding = sequence_length - len(sentence)
#         new_sentence = sentence + [padding_word] * num_padding
#         padded_sentences.append(new_sentence)
#     return padded_sentences
#
#
# def build_vocab(sentences):
#     """
#     Builds a vocabulary mapping from word to index based on the sentences.
#     Returns vocabulary mapping and inverse vocabulary mapping.
#     """
#     # Build vocabulary
#     word_counts = Counter(itertools.chain(*sentences))
#     # Mapping from index to word
#     vocabulary_inv = [x[0] for x in word_counts.most_common()]
#     vocabulary_inv = list(sorted(vocabulary_inv))
#     # Mapping from word to index
#     vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
#     return [vocabulary, vocabulary_inv]
#
#
# def build_input_data(sentences, labels, vocabulary):
#     """
#     Maps sentencs and labels to vectors based on a vocabulary.
#     """
#     def get_index_of_voca(vocabulary, word):
#         try:
#             return vocabulary[word]
#         except:
#             return vocabulary["<PAD/>"]
#
#     x = np.array([[get_index_of_voca(vocabulary,word) for word in sentence] for sentence in sentences])
#     y = np.array(labels)
#     return [x, y]
#
#
# def build_input_data_with_w2v(sentences, labels, w2vmodel):
#     """
#     Maps sentencs and labels to vectors based on a vocabulary.
#     """
#     def get_index_of_voca(model, word):
#         try:
#             return model[word]
#         except:
#             return np.array([np.float32(0.0)]*400)
#
#     x = np.array([[get_index_of_voca(w2vmodel,word) for word in sentence] for sentence in sentences])
#     y = np.array(labels)
#     return [x, y]
#
#
# def load_data(dataset, w2vmodel, padlen=None):
#     """
#     Loads and preprocessed data for the MR dataset.
#     Returns input vectors, labels, vocabulary, and inverse vocabulary.
#     """
#     # Load and preprocess data
#     sentences, labels = load_data_and_labels(dataset)
#     sentences_padded = pad_sentences(sentences, padlen)
#
#     x, y = build_input_data_with_w2v(sentences_padded, labels, w2vmodel)
#     return [x, y]
#
#
# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     """
#     Generates a batch iterator for a dataset.
#     """
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int(len(data)/batch_size) + 1
#     for epoch in range(num_epochs):
#         # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batch_num * batch_size
#             end_index = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_index:end_index]
#
#
#
#
# class DataSet(object):
#     def __init__(self,
#                 inputs,
#                 labels):
#
#         assert inputs.shape[0] == labels.shape[0], (
#             'inputs.shape: %s labels.shape: %s' % (inputs.shape, labels.shape))
#         self._num_examples = inputs.shape[0]
#
#         self._inputs = inputs
#         self._labels = labels
#         self._epochs_completed = 0
#         self._index_in_epoch = 0
#
#     @property
#     def inputs(self):
#         return self._inputs
#
#     @property
#     def labels(self):
#         return self._labels
#
#     @property
#     def num_examples(self):
#         return self._num_examples
#
#     @property
#     def epochs_completed(self):
#         return self._epochs_completed
#
#     def next_batch(self, batch_size):
#         """Return all examples if batch_size==0."""
#         if batch_size==0:
#             start = self._index_in_epoch
#             end = self._num_examples
#             return self._inputs[start:end], self._labels[start:end]
#
#         """Return the next `batch_size` examples from this data set."""
#         start = self._index_in_epoch
#         self._index_in_epoch += batch_size
#         if self._index_in_epoch > self._num_examples:
#             # Finished epoch
#             self._epochs_completed += 1
#             # Shuffle the data
#             perm = numpy.arange(self._num_examples)
#             numpy.random.shuffle(perm)
#             self._inputs = self._inputs[perm]
#             self._labels = self._labels[perm]
#             # Start next epoch
#             start = 0
#             self._index_in_epoch = batch_size
#             assert batch_size <= self._num_examples
#         end = self._index_in_epoch
#         return self._inputs[start:end], self._labels[start:end]
#
#
#
#
# def read_tweets(train_dir, selected_features_data):
#     local_file = base.maybe_download(selected_features_data, train_dir,
#                                      BASE_URL + selected_features_data)
#
#     with open(local_file, 'rb') as handle:
#
#
#
#
#     train = DataSet(train_x, train_labels)
#     validation = DataSet(validation_x, validation_labels)
#     test = DataSet(test_x, test_labels)
#
#     return base.Datasets(train=train, validation=validation, test=test)
#
#
#
# def read_data_sets(train_dir, selected_features_data, norm_method, n_class, dtype=dtypes.float32):
#     # selected_features_data = 'cancer_dataset_sel_nonneg_shuffle.pickle.pickle'
#     # selected_features_data = 'cancer_dataset_sel_shuffle.pickle'
#     # selected_features_data = 'cancer_dataset_sel_tiny.pickle'
#
#     VALIDATION_RATIO = 0.1
#     TEST_RATIO = 0.2
#
#     # maybe_download(filename, work_directory, source_url):
#     local_file = base.maybe_download(selected_features_data, train_dir,
#                                      BASE_URL + selected_features_data)
#
#     with open(local_file, 'rb') as handle:
#         X_shuffle = pickle.load(handle)
#         y_shuffle = pickle.load(handle)
#
#         X_shuffle = numpy.array(X_shuffle)
#         y_shuffle = numpy.eye(n_class)[numpy.array(y_shuffle)]
#
#     VALIDATION_SIZE = int(X_shuffle.shape[0]*VALIDATION_RATIO)
#     TEST_SIZE = int(X_shuffle.shape[0] * TEST_RATIO)
#     TRAIN_SIZE = X_shuffle.shape[0] - (VALIDATION_SIZE+TEST_SIZE)
#
#     train_x = X_shuffle[0:TRAIN_SIZE]
#     train_labels = y_shuffle[0:TRAIN_SIZE]
#
#     validation_x = X_shuffle[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
#     validation_labels = y_shuffle[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
#
#     test_x = X_shuffle[TRAIN_SIZE + VALIDATION_SIZE:]
#     test_labels = y_shuffle[TRAIN_SIZE + VALIDATION_SIZE:]
#
#     if norm_method == 'max':
#         train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#         print('norm max')
#
#     elif norm_method == 'maxall':
#         train_x, validation_x, test_x = maxall_normalization(train_x, validation_x, test_x)
#         print('norm maxall')
#
#     elif norm_method == 'sum':
#         train_x, validation_x, test_x = sum_normalization(train_x, validation_x, test_x)
#         print('norm sum')
#
#     elif norm_method == 'none':
#         print('no normalization')
#
#     else:
#         train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#         print('norm max')
#
#     train = DataSet(train_x, train_labels)
#     validation = DataSet(validation_x, validation_labels)
#     test = DataSet(test_x, test_labels)
#
#     return base.Datasets(train=train, validation=validation, test=test)
#
#
# def read_data_sets_loocv_with_type(train_dir, selected_features_data, norm_method, n_class, type, dtype=dtypes.float32):
#     VALIDATION_RATIO = 0
#     TEST_RATIO = 0
#
#     # maybe_download(filename, work_directory, source_url):
#     local_file = base.maybe_download(selected_features_data, train_dir,
#                                      BASE_URL + selected_features_data)
#
#     with open(local_file, 'rb') as handle:
#         X_shuffle = pickle.load(handle)
#         y_shuffle = pickle.load(handle)
#         y_type_shuffle = pickle.load(handle)
#
#         X_shuffle = numpy.array(X_shuffle)
#         # y_shuffle = numpy.eye(n_class)[numpy.array(y_shuffle)]
#
#     selected_type = [index for index,value in enumerate(y_type_shuffle) if value%100 == type]
#     X_shuffle = X_shuffle[selected_type]
#     y_shuffle = y_shuffle[selected_type]
#
#     TRAIN_SIZE = X_shuffle.shape[0]
#
#     train_x = X_shuffle[0:TRAIN_SIZE]
#     train_labels = y_shuffle[0:TRAIN_SIZE]
#
#     validation_x = X_shuffle[-1:]
#     validation_labels = y_shuffle[-1:]
#
#     test_x = X_shuffle[-1:]
#     test_labels = y_shuffle[-1:]
#
#     # if norm_method == 'max':
#     #     train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#     #     print('norm max')
#     #
#     # elif norm_method == 'maxall':
#     #     train_x, validation_x, test_x = maxall_normalization(train_x, validation_x, test_x)
#     #     print('norm maxall')
#     #
#     # elif norm_method == 'sum':
#     #     train_x, validation_x, test_x = sum_normalization(train_x, validation_x, test_x)
#     #     print('norm sum')
#     #
#     # elif norm_method == 'none':
#     #     print('no normalization')
#     #
#     # else:
#     #     train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#     #     print('norm max')
#
#
#     train = DataSet(train_x, train_labels)
#     validation = DataSet(validation_x, validation_labels)
#     test = DataSet(test_x, test_labels)
#
#
#
#
#     return base.Datasets(train=train, validation=validation, test=test)
#
#
# def read_data_sets_with_type(train_dir, selected_features_data, norm_method, n_class, type, dtype=dtypes.float32):
#
#     VALIDATION_RATIO = 0.1
#     TEST_RATIO = 0.2
#
#     # maybe_download(filename, work_directory, source_url):
#     local_file = base.maybe_download(selected_features_data, train_dir,
#                                      BASE_URL + selected_features_data)
#
#     with open(local_file, 'rb') as handle:
#         X_shuffle = pickle.load(handle)
#         y_shuffle = pickle.load(handle)
#         y_type_shuffle = pickle.load(handle)
#
#         X_shuffle = numpy.array(X_shuffle)
#         y_shuffle = numpy.eye(n_class)[numpy.array(y_shuffle)]
#
#     selected_type = [index for index,value in enumerate(y_type_shuffle) if value%100 == type]
#     X_shuffle = X_shuffle[selected_type]
#     y_shuffle = y_shuffle[selected_type]
#
#
#     VALIDATION_SIZE = int(X_shuffle.shape[0]*VALIDATION_RATIO)
#     TEST_SIZE = int(X_shuffle.shape[0] * TEST_RATIO)
#     TRAIN_SIZE = X_shuffle.shape[0] - (VALIDATION_SIZE+TEST_SIZE)
#
#     train_x = X_shuffle[0:TRAIN_SIZE]
#     train_labels = y_shuffle[0:TRAIN_SIZE]
#
#     validation_x = X_shuffle[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
#     validation_labels = y_shuffle[TRAIN_SIZE:TRAIN_SIZE+VALIDATION_SIZE]
#
#     test_x = X_shuffle[TRAIN_SIZE + VALIDATION_SIZE:]
#     test_labels = y_shuffle[TRAIN_SIZE + VALIDATION_SIZE:]
#
#     if norm_method == 'max':
#         train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#         print('norm max')
#
#     elif norm_method == 'maxall':
#         train_x, validation_x, test_x = maxall_normalization(train_x, validation_x, test_x)
#         print('norm maxall')
#
#     elif norm_method == 'sum':
#         train_x, validation_x, test_x = sum_normalization(train_x, validation_x, test_x)
#         print('norm sum')
#
#     elif norm_method == 'none':
#         print('no normalization')
#
#     else:
#         train_x, validation_x, test_x = max_normalization(train_x, validation_x, test_x)
#         print('norm max')
#
#     train = DataSet(train_x, train_labels)
#     validation = DataSet(validation_x, validation_labels)
#     test = DataSet(test_x, test_labels)
#
#     return base.Datasets(train=train, validation=validation, test=test)