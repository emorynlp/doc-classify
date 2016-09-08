import numpy as np
import re
import itertools
from collections import Counter



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

def load_data_trainable(dataset, rottenTomato=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    if rottenTomato:
        template_txt = '../data/rt-data-nlp4jtok/%s'
    else:
        template_txt = '../data/tweets/txt/%s'

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

def load_data_and_labels(dataset, rottenTomato=False):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    if rottenTomato:
        template_txt = '../data/rt-data-nlp4jtok/%s'
    else:
        template_txt = '../data/tweets/txt/%s'

    pathtxt = template_txt % dataset

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

def load_data(dataset, w2vmodel, lexiconModel, padlen=None, rottenTomato=False):
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
    return [x, y, x_lex]

    # x, y = build_input_data_with_w2v(sentences_padded, labels, w2vmodel)
    # return [x, y]


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
