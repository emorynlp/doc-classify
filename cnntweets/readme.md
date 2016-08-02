## Deep Ensemble for Sentiment Analysis


## Installation
* clone the repository

```bash
git clone git@github.com:bgshin/cnntweets.git
```
* make virtual env

```bash
mkvirtualenv sent
```


* Dependencies

```bash
pip install -r requirements.txt
```

* Python 2.7
* requirements
	* boto==2.40.0
	* bz2file==0.98
	* gensim==0.12.4
	* numpy==1.11.0
	* protobuf==3.0.0b2
	* requests==2.10.0
	* scipy==0.17.1
	* six==1.10.0
	* smart-open==1.3.3
	* tensorflow==0.8.0




## Usage

* **WITHOUT** pre-trained w2v
	* Train 
	
		```bash
		cd cnn
		nohup python cnn_train.py > out.txt &
		```
	
	* Test 
		* Modify cnn/cnn_test.py

			```python
			savepath = 'model_path/model-xxxx'
			```
		* Run test script
	
			```bash
			cd cnn
			python cnn_test.py
			```

* **WITH** pre-trained w2v
	* Download and extract the compressed file to have the pre-trained w2v bin file
		* [download:link1](http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz)
	* Modify w2v_cnn/cnn_train.py

		```python
		model_path = 'path_to_w2v_bin/word2vec_twitter_model.bin'
		```

	* Train 
	
		```bash
		cd w2v_cnn
		nohup python cnn_train.py > out.txt &
		```
	
	* Test 
		* Modify w2v_cnn/cnn_test.py

			```python
			savepath = 'model_path/model-xxxx'
			```
		* Run test script
	
			```bash
			cd w2v_cnn
			python cnn_test.py
			```


## Dataset
### Semeval 2016
* Dev (semeval16\_T4A\_devtest\_npo)
	* number of data: 1588
* Tst (semeval16\_T4A\_test\_npo)
	* number of data: 20632
* Trn (semeval13\_T2B\_16T4A\_train\_dev\_npo)
	* number of data: 15385

* Data files
	* semeval13\_T2B\_16T4A\_train\_dev\_devtest\_npo - 1588+15385 = 16973
	* **semeval16\_T4A\_devtest\_npo = 1588**
	* **semeval13\_T2B\_16T4A\_train\_dev\_npo = 15385**
	* semeval16\_T4A\_dev\_npo = 1595
	* **semeval16\_T4A\_test\_npo = 20632**
	* semeval16\_T4A\_train\_npo = 4796

* Format of data (TAB separated) 

	| no | sentiment | sentences |
	|----|-----------|-----------|
	| 1 | objective | I may be the ... |	
	| 2 | positive | TGIF folks! ... |	


	
## Preprocessing
* Label definition
	* 'objective': [0, 1, 0], 1
	* 'positive': [0, 0, 1], 2
	* 'negative': [1, 0, 0], 0


## Reference

### Pre-trained Word2vec done by Fréderic Godin
* Trained on 400 million tweets
* Resources
	* [download:link](http://yuca.test.iminds.be:8900/fgodin/downloads/word2vec_twitter_model.tar.gz)
	* [reference1:github](https://github.com/FredericGodin/DynamicCNN)
	* [reference2:intro](https://groups.google.com/forum/#!topic/word2vec-toolkit/qFwm5p2qWqM)
