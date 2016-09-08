import numpy as np
import os
import re
from cnntweets.utils.butils import Timer
import pickle

def load_lexicon_unigram(lexdim):
    if lexdim==6:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'HS-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'Maxdiff-Twitter-Lexicon_0to1.txt': [0.5],
                              'S140-AFFLEX-NEGLEX-unigrams.txt': [0],
                              'unigrams-pmilexicon.txt': [0],
                              'unigrams-pmilexicon_sentiment_140.txt': [0]}

    elif lexdim == 2:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0]}

    elif lexdim == 4:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt': [0],
                              'unigrams-pmilexicon.txt': [0, 0, 0]}

    elif lexdim == 15:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}
    else:
        default_vector_dic = {'EverythingUnigramsPMIHS.txt':[0],
                          'HS-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'Maxdiff-Twitter-Lexicon_0to1.txt':[0.5],
                          'S140-AFFLEX-NEGLEX-unigrams.txt':[0,0,0],
                          'unigrams-pmilexicon.txt':[0,0,0],
                          'unigrams-pmilexicon_sentiment_140.txt':[0,0,0],
                          'BL.txt': [0]}

    file_path = ["../../data/lexicon_data/"+files for files in os.listdir("../../data/lexicon_data") if files.endswith(".txt")]
    if lexdim == 2 or lexdim == 4:
        raw_model = [dict() for x in range(2)]
        norm_model = [dict() for x in range(2)]
        file_path = ['../../data/lexicon_data/EverythingUnigramsPMIHS.txt', '../../data/lexicon_data/unigrams-pmilexicon.txt']
    else:
        raw_model = [dict() for x in range(len(file_path))]
        norm_model = [dict() for x in range(len(file_path))]

    for index, each_model in enumerate(raw_model):
        data_type = file_path[index].replace("../../data/lexicon_data/", "")
        # if lexdim == 2 or lexdim == 4:
        #     if data_type not in ['EverythingUnigramsPMIHS.txt', 'unigrams-pmilexicon.txt']:
        #         continue

        default_vector = default_vector_dic[data_type]

        # print data_type, default_vector
        raw_model[index]["<PAD/>"] = default_vector

        with open(file_path[index], 'r') as document:
            for line in document:
                line_token = re.split(r'\t', line)

                data_vec=[]
                key=''

                if lexdim == 2 or lexdim == 6:
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk

                        elif idx == 1:
                            data_vec.append(float(tk))

                        else:
                            continue

                else: # 4 or 14
                    for idx, tk in enumerate(line_token):
                        if idx == 0:
                            key = tk
                        else:
                            try:
                                data_vec.append(float(tk))
                            except:
                                pass


                assert(key != '')
                each_model[key] = data_vec

    for index, each_model in enumerate(norm_model):
    # for m in range(len(raw_model)):
        values = np.array(raw_model[index].values())
        new_val = np.copy(values)


        #print 'model %d' % index
        for i in range(len(raw_model[index].values()[0])):
            pos = np.max(values, axis=0)[i]
            neg = np.min(values, axis=0)[i]
            mmax = max(abs(pos), abs(neg))
            #print pos, neg, mmax

            new_val[:, i] = values[:, i] / mmax

        keys = raw_model[index].keys()
        dictionary = dict(zip(keys, new_val))

        norm_model[index] = dictionary


    return norm_model, raw_model


with Timer("lex"):
    norm_model, raw_model = load_lexicon_unigram(15)

with Timer("saving pickle for 15dim lexicon"):
    with open('../../data/lexicon_data/lex15.pickle', 'wb') as handle:
        pickle.dump(norm_model, handle)
