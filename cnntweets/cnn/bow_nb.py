from collections import Counter
import csv
import math


template_txt = '../data/tweets/txt/%s'
pathtxt = template_txt % 'dev'

def getDocumentFrequency(df, term):
    if term in df:
        return df[term] + 1
    else:
        return 1

def getTFIDF(tf, df, dc):
    return tf * math.log(float(dc) / df)


def getTFIDFs(tf, df, dc):
    return [{k: getTFIDF(v, getDocumentFrequency(df, k), dc) for (k, v) in d.items()} for d in tf]


nDoc=0
tf_corpus=Counter()
with open(pathtxt, 'rt') as csvfile:
    rows = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    sentiment='notdet'

    tfs=[]

    for line in rows:
        nDoc=nDoc+1
        tf=Counter()
        tf_corpus.update(map(lambda x: x.lower(), line[2].split()))
        tf.update(map(lambda x: x.lower(), line[2].split()))
        tfs.append(tf)


df = Counter()
for d in tfs:
    df.update(d.keys())


tfidf=getTFIDFs(tfs,df,nDoc)

print nDoc

# tf-idf: tf/log(idf+1)




# print trnTF

