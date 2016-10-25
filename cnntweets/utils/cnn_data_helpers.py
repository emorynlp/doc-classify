import numpy as np
import re
import itertools
from collections import Counter
import os
from utils.word2vecReader import Word2Vec


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_test_data_and_labels(dataset, rottenTomato=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    if rottenTomato:
        template_txt = '../data/rt-data-nlp4jtok/%s'
    else:
        template_txt = '../data/tweets/%s'

    pathtxt = template_txt % dataset

    x_text=[line.split('\t')[0] for line in open(pathtxt, "r").readlines()]
    x_text = [s.split(" ") for s in x_text]

    return x_text


def load_data_and_labels(dataset, rottenTomato=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    pathtxt = dataset

    x_text=[line.split('\t')[2] for line in open(pathtxt, "r").readlines()]
    x_text = [s.split(" ") for s in x_text]

    y = []
    if rottenTomato:
        for line in open(pathtxt, "r").readlines():
            senti=line.split('\t')[1]
            if  senti == 'neutral':
                y.append([0, 0, 1, 0, 0])

            elif senti == 'positive':
                y.append([0, 0, 0, 1, 0])
            elif senti == 'very_positive':
                y.append([0, 0, 0, 0 ,1])
            elif senti == 'negative':
                y.append([0, 1, 0, 0 ,0])
            elif senti == 'very_negative':
                y.append([1, 0, 0, 0 ,0])

    else:
        for line in open(pathtxt, "r").readlines():
            senti=line.split('\t')[1]
            if  senti == 'objective':
                y.append([0, 1, 0])

            elif senti == 'positive':
                y.append([0, 0, 1])

            else:  # negative
                y.append([1, 0, 0])

    return [x_text, y]


def load_data_trainable(dataset, rottenTomato=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    if rottenTomato:
        template_txt = '../data/rt-data-nlp4jtok/%s'
    else:
        template_txt = '../data/tweets/%s'

    pathtxt = template_txt % dataset

    x_text=[line.split('\t')[2] for line in open(pathtxt, "r").readlines()]
    x_text = [sent for sent in x_text]

    y = []
    if rottenTomato:
        for line in open(pathtxt, "r").readlines():
            senti=line.split('\t')[1]
            if  senti == 'neutral':
                y.append([0, 0, 1, 0, 0])

            elif senti == 'positive':
                y.append([0, 0, 0, 1, 0])
            elif senti == 'very_positive':
                y.append([0, 0, 0, 0 ,1])
            elif senti == 'negative':
                y.append([0, 1, 0, 0 ,0])
            elif senti == 'very_negative':
                y.append([1, 0, 0, 0 ,0])

    else:
        for line in open(pathtxt, "r").readlines():
            senti=line.split('\t')[1]
            if  senti == 'objective':
                y.append([0, 1, 0])

            elif senti == 'positive':
                y.append([0, 0, 1])

            else:  # negative
                y.append([1, 0, 0])

    return [x_text, y]



def pad_sentences(sentences, padlen, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    if padlen==None:
        sequence_length = max(len(x) for x in sentences)
    else:
        sequence_length=padlen

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_lex_data(sentences, lexiconModel, padlen):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    def get_index_of_vocab_lex(lexiconModel, word):
        lexiconList = np.empty([0, 1])
        for index, eachModel in enumerate(lexiconModel):
            if word in eachModel:
                temp = np.array(np.float32(eachModel[word]))
            else:
                temp = np.array(np.float32(eachModel["<PAD/>"]))
            lexiconList = np.append(lexiconList, temp)

        if len(lexiconList) > 15:
            print len(lexiconList)
            print '======================over 15======================'
        return lexiconList

    sentences = [ sentence.split(" ") for sentence in sentences]
    sentences_padded = pad_sentences(sentences, padlen)
    x_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in sentences_padded])
    return x_lex

def build_w2v_data(dataset, w2vmodel, padlen):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    sentences, labels = load_data_and_labels(dataset)
    sentences_padded = pad_sentences(sentences, padlen)

    w2v_dim = len(w2vmodel["a"])

    def get_index_of_voca(model, word):
        try:
            return model[word]
        except:
            return np.array([np.float32(0.0)]*w2v_dim)


    x = np.array([[get_index_of_voca(w2vmodel,word) for word in sentence] for sentence in sentences_padded])
    y = np.array(labels)
    return [x, y]


def build_input_data_with_w2v(sentences, labels, w2vmodel, lexiconModel):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    w2v_dim = len(w2vmodel["a"])

    def get_index_of_voca(model, word):
        try:
            return model[word]
        except:
            return np.array([np.float32(0.0)]*w2v_dim)

    def get_index_of_vocab_lex(lexiconModel, word):
        lexiconList = np.empty([0, 1])
        for index, eachModel in enumerate(lexiconModel):
            if word in eachModel:
                temp = np.array(np.float32(eachModel[word]))
            else:
                temp = np.array(np.float32(eachModel["<PAD/>"]))
            lexiconList = np.append(lexiconList, temp)

        if len(lexiconList) > 15:
            print len(lexiconList)
            print '======================over 15======================'
        return lexiconList

    x = np.array([[get_index_of_voca(w2vmodel,word) for word in sentence] for sentence in sentences])
    x_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y, x_lex]

def build_test_input_data_with_w2v(sentences, w2vmodel, lexiconModel):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    w2v_dim = len(w2vmodel["a"])

    def get_index_of_voca(model, word):
        try:
            return model[word]
        except:
            return np.array([np.float32(0.0)]*w2v_dim)

    def get_index_of_vocab_lex(lexiconModel, word):
        lexiconList = np.empty([0, 1])
        for index, eachModel in enumerate(lexiconModel):
            if word in eachModel:
                temp = np.array(np.float32(eachModel[word]))
            else:
                temp = np.array(np.float32(eachModel["<PAD/>"]))
            lexiconList = np.append(lexiconList, temp)

        if len(lexiconList) > 15:
            print len(lexiconList)
            print '======================over 15======================'
        return lexiconList

    x = np.array([[get_index_of_voca(w2vmodel,word) for word in sentence] for sentence in sentences])
    x_lex = np.array([[get_index_of_vocab_lex(lexiconModel, word) for word in sentence] for sentence in sentences])
    return [x, x_lex]

def load_data(dataset, w2vmodel, lexiconModel, padlen=None, rottenTomato=False, multichannel=False):
# def load_data(dataset, w2vmodel, padlen=None):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    if rottenTomato:
        sentences, labels = load_data_and_labels(dataset, True)
        sentences_padded = pad_sentences(sentences, padlen)
    else:
        sentences, labels = load_data_and_labels(dataset)
        sentences_padded = pad_sentences(sentences, padlen)

    x, y , x_lex = build_input_data_with_w2v(sentences_padded, labels, w2vmodel, lexiconModel)


    if multichannel==True:
        w2vdim = x[0].shape[1]
        lexdim = x_lex[0].shape[1]
        npad = ((0, 0), (0, w2vdim - lexdim))

        new_x_batch = []
        for idx, xlex in enumerate(x_lex):
            xlex_padded = np.pad(xlex, pad_width=npad, mode='constant', constant_values=0)
            new_x_batch.append(np.concatenate((x[idx][..., np.newaxis], xlex_padded[..., np.newaxis]), axis=2))

        x_fat = np.array(new_x_batch)
        return [x, y, x_lex, x_fat]

    return [x, y, x_lex, None]

    # x, y = build_input_data_with_w2v(sentences_padded, labels, w2vmodel)
    # return [x, y]

def load_test_data(dataset, w2vmodel, lexiconModel, padlen=None, rottenTomato=False, multichannel=False):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    if rottenTomato:
        sentences, labels = load_data_and_labels(dataset, True)
        sentences_padded = pad_sentences(sentences, padlen)
    else:
        sentences = load_test_data_and_labels(dataset)
        sentences_padded = pad_sentences(sentences, padlen)

    x, x_lex = build_test_input_data_with_w2v(sentences_padded, w2vmodel, lexiconModel)


    if multichannel==True:
        w2vdim = x[0].shape[1]
        lexdim = x_lex[0].shape[1]
        npad = ((0, 0), (0, w2vdim - lexdim))

        new_x_batch = []
        for idx, xlex in enumerate(x_lex):
            xlex_padded = np.pad(xlex, pad_width=npad, mode='constant', constant_values=0)
            new_x_batch.append(np.concatenate((x[idx][..., np.newaxis], xlex_padded[..., np.newaxis]), axis=2))

        x_fat = np.array(new_x_batch)
        return [x, x_lex, x_fat]

    return [x, x_lex, None]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    np.random.seed(3)  # FIX RANDOM SEED
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_w2v(w2vdim, simple_run=True, source="twitter"):
    if simple_run:
        return {'a': np.array([np.float32(0.0)] * w2vdim)}

    else:
        if source == "twitter":
            model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
        elif source == "amazon":
            model_path = '../data/emory_w2v/w2v-%d-%s.bin' % (w2vdim, source)

        model = Word2Vec.load_word2vec_format(model_path, binary=True)
        print("The vocabulary size is: " + str(len(model.vocab)))

        return model

def load_w2v_withpath(model_path):
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model


def load_lexicon_unigram(file_path_list):
    default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
                          'unigrams-pmilexicon.txt': [0, 0, 0],
                          'unigrams-pmilexicon_sentiment_140.txt': [0, 0, 0],
                          'BL.txt': [0]}


    raw_model = [dict() for x in range(len(file_path_list))]
    norm_model = [dict() for x in range(len(file_path_list))]

    for index, each_model in enumerate(raw_model):
        data_type = file_path_list[index].replace("../data/lexicon_data/", "")
        print file_path_list[index]
        print data_type
        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector

        with open(file_path_list[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec = []
                key = ''


                for idx, tk in enumerate(line_token):
                    if idx == 0:
                        key = tk
                    else:
                        try:
                            data_vec.append(float(tk))
                        except:
                            pass

                assert (key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
        # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)

        # print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            # print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[index].keys()
        dictionary = dict(zip(keys, new_val))

        norm_model[index] = dictionary

    return norm_model, raw_model