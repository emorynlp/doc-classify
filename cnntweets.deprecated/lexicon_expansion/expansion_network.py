from cnntweets.utils.butils import Timer
from cnntweets.train_model import load_w2v2
import re
import numpy as np
import pickle

# default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
#                       'HS-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
#                       'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
#                       'S140-AFFLEX-NEGLEX-unigrams.txt': [0, 0, 0],
#                       'unigrams-pmilexicon.txt': [0, 0, 0],
#                       'unigrams-pmilexicon_sentiment_140.txt': [0, 0, 0],
#                       'BL.txt': [0]}


def load_lexicon_unigram(lexfile):
    lexdim_dic = {'EverythingUnigramsPMIHS.txt': 1,
                  'HS-AFFLEX-NEGLEX-unigrams.txt': 3,
                  'Maxdiff-Twitter-Lexicon_0to1.txt': 1,
                  'S140-AFFLEX-NEGLEX-unigrams.txt': 3,
                  'unigrams-pmilexicon.txt': 3,
                  'unigrams-pmilexicon_sentiment_140.txt': 3,
                  'BL.txt': 1}

    lexdim = lexdim_dic[lexfile]
    lexfile = "../../data/lexicon_data/" + lexfile

    raw_model = dict()
    norm_model = dict()
    with open(lexfile, 'r') as document:
        for line in document:
            line_token = re.split(r'\t', line)

            data_vec=[]
            key=''

            if lexdim == 1:
                for idx, tk in enumerate(line_token):
                    if idx == 0:
                        key = tk

                    elif idx == 1:
                        data_vec.append(float(tk))

                    else:
                        continue

            else: # 3
                for idx, tk in enumerate(line_token):
                    if idx == 0:
                        key = tk
                    else:
                        try:
                            data_vec.append(float(tk))
                        except:
                            pass


            assert(key != '')
            raw_model[key] = data_vec



    values = np.array(raw_model.values())
    new_val = np.copy(values)


    #print 'model %d' % index
    for i in range(len(raw_model.values()[0])):
        pos = np.max(values, axis=0)[i]
        neg = np.min(values, axis=0)[i]
        mmax = max(abs(pos), abs(neg))
        #print pos, neg, mmax

        new_val[:, i] = values[:, i] / mmax

    keys = raw_model.keys()
    dictionary = dict(zip(keys, new_val))

    norm_model = dictionary


    return norm_model, raw_model


def get_train_data(w2vmodel, lexfile, simple_run = True):
    print 'lexfile(%s)' % (lexfile)
    if simple_run == True:
        print '======================================[simple_run]======================================'

    max_len = 60




    with Timer("lex"):
        norm_model, raw_model = load_lexicon_unigram(lexfile)

    words = []
    data_x = []
    data_y = []
    for word, lex in norm_model.iteritems():
        try:
            test = w2vmodel[word]
        except:
            continue

        words.append(word)
        data_x.append(w2vmodel[word])
        data_y.append(lex)


    x = np.array(data_x)
    y = np.array(data_y)

    print 'data size for %s is %d' % (lexfile, len(y))

    with Timer("saving pickle for %s" % lexfile):
        with open('../../data/le/%s.pickle' % lexfile.replace('.txt', ''), 'wb') as handle:
            pickle.dump(x, handle)
            pickle.dump(y, handle)



if __name__ == "__main__":
    lexfile_list = ['EverythingUnigramsPMIHS.txt',
                    'HS-AFFLEX-NEGLEX-unigrams.txt',
                    'Maxdiff-Twitter-Lexicon_0to1.txt',
                    'S140-AFFLEX-NEGLEX-unigrams.txt',
                    'unigrams-pmilexicon.txt',
                    'unigrams-pmilexicon_sentiment_140.txt',
                    'BL.txt']

    w2vdim = 400

    with Timer("w2v"):
        w2vmodel = load_w2v2(w2vdim, simple_run=False, base_path='../../data/emory_w2v/')

    for lexfile in lexfile_list:
        get_train_data(w2vmodel, lexfile, simple_run = False)