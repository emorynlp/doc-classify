import os
import time
import re
import numpy as np

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):

        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


def load_lexicon_unigram():
    default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                      'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                      'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                      'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                      'unigrams-pmilexicon.txt':[0,0,0],
                      'unigrams-pmilexicon_sentiment_140.txt':[0,0,0]}

    file_path = ["../data/lexicon_data/"+files for files in os.listdir("../data/lexicon_data") if files.endswith(".txt")]
    raw_model = [dict() for x in range(len(file_path))]
    norm_model = [dict() for x in range(len(file_path))]
    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../data/lexicon_data/", "")
        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector


        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''
                for idx, tk in enumerate(line_token):
                    if idx == 0:
                        key = tk
                    else:
                        data_vec.append(float(tk))

                assert(key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
    # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)

        print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[1].keys()
        dictionary = dict(zip(keys, new_val))
        norm_model[index] = dictionary

        data_type = file_path[index].replace("../data/lexicon_data/", "")
        default_vector = default_vector_dic[data_type]

        dictionary["<PAD/>"] = default_vector
        # models.append(dictionary)

    return norm_model, raw_model

# models=[]
with Timer("lex"):
    norm_model, model_raw = load_lexicon_unigram()


print norm_model[1].values()[0:2]
print model_raw[1].values()[0:2]
print [xx["<PAD/>"] for xx in model_raw]
print [xx["<PAD/>"] for xx in norm_model]

print 'hi'

