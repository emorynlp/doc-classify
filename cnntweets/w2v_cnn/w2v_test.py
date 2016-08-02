from word2vecReader import Word2Vec
import time


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):

        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)


model_path='/Users/bong/works/data/word2vec_twitter_model/word2vec_twitter_model.bin'
with Timer("load w2v"):
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))
    # model = Word2Vec.load(model_path)


print model['apple']

