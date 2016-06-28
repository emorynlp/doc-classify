# doc-classify
Document classification.

<em>This code is based on http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/ </em>

<b>REQUIREMENTS</b> <br>
tensorflow
python 2.7
gensim
... (whatever required when you run "python cnn_train.py")

<b>DIRECTION</b> <br>
1. After installing all the requirements, clone this repository.
2. Download Twitter W2V Bin File: http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz. Place it inside doc-classify foler.
3. Create a folder called "txt" and put your training, dev, test data inside this folder.
4. Run python cnn_train.py to train.
5. Run python cnn_test.py to evaluate using test set

