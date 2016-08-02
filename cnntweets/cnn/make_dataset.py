import csv
from os import listdir
from os.path import isfile, join



template_nlp = 'data/tweets/nlp/%s/'
template_txt = 'data/tweets/txt/%s'


trnpath = 'semeval13_T2B_16T4A_train_dev_npo'
tstpath = 'semeval16_T4A_test_npo'
devpath = 'semeval16_T4A_devtest_npo'

def make_data(pathnlp, pathtxt):
    onlyfiles = [f for f in listdir(pathnlp) if isfile(join(pathnlp, f))]

    with open(pathtxt, 'wt') as tw:
        num=0
        lastnum=0
        for f in onlyfiles:
            num=num+1

            if f.endswith('nlp'):
                # if f=='semeval13_T2B_16T4A_train_dev_npo_100.nlp':
                #     print f
                with open(join(pathnlp, f), 'rb') as csvfile:
                    rows = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
                    word_list=[]
                    lemma_list=[]
                    sentiment='notdet'
                    for row in rows:
                        # if f == 'semeval13_T2B_16T4A_train_dev_npo_100.nlp':
                        #     continue
                        if len(row)==0:
                            continue

                        if (row[4].endswith('objective')):
                            sentiment = 'objective'
                        if (row[4].endswith('positive')):
                            sentiment = 'positive'
                        if (row[4].endswith('negative')):
                            sentiment = 'negative'

                        # if row[2]=="#hlink#":
                        #     continue

                        word_list.append(row[1])
                        lemma_list.append(row[2])

                    sentence = ' '.join(word_list)
                    lemma_sentence = ' '.join(lemma_list)

                    if sentiment is 'notdet':
                        continue
                    assert (sentiment is not 'notdet')

                if lastnum+1!=num:
                    print 'wrong'
                data_template = '%d\t%s\t%s\t%s\n' % (num, sentiment, sentence, lemma_sentence)
                lastnum=num
                tw.write(data_template)


pathnlp = template_nlp % tstpath
pathtxt = template_txt % 'tst'
make_data(pathnlp, pathtxt)

pathnlp = template_nlp % trnpath
pathtxt = template_txt % 'trn'
make_data(pathnlp, pathtxt)

pathnlp = template_nlp % devpath
pathtxt = template_txt % 'dev'
make_data(pathnlp, pathtxt)
