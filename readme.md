## Emory Sentiment Analysis


## Installation
### clone the repository

```bash
git@github.com:emorynlp/doc-classify.git
```
* make virtual env

```bash
mkvirtualenv sent
```


### Dependencies

```bash
pip install -r requirements.txt
```

* Python 2.7
* requirements
	* boto==2.42.0
	* bz2file==0.98
	* funcsigs==1.0.2
	* gensim==0.13.2
	* mock==2.0.0
	* numpy==1.11.2
	* pbr==1.10.0
	* protobuf==3.0.0b2
	* requests==2.11.1
	* scipy==0.18.1
	* six==1.10.0
	* smart-open==1.3.5
	* tensorflow==0.10.0rc0


### Required files
* Please contact us to obtain the following files
* word2vec
	* filename
	
```
data/emory_w2v/w2v-400.bin
```
* lexicon_data
	* filenames
	
```
data/lexicon_data/BL.txt
data/lexicon_data/EverythingUnigramsPMIHS.txt
data/lexicon_data/HS-AFFLEX-NEGLEX-unigrams.txt
data/lexicon_data/Maxdiff-Twitter-Lexicon_0to1.txt
data/lexicon_data/S140-AFFLEX-NEGLEX-unigrams.txt
data/lexicon_data/unigrams-pmilexicon_sentiment_140.txt
data/lexicon_data/unigrams-pmilexicon.txt
```

* trained_models
	* filenames
	
```
data/trained_models/model-w2v
data/trained_models/model-w2vlexatt
```
	
	


## Usage
### Train 
* Input file format - **note that the text should be tokenized**

| line number | sentiment | tokenized tweet        |
|-------------|-----------|------------------------|
| 1           | object    | I may be the only one… |
| 2           | negative  | If Scotland woke…      |
	
```bash
cd cnntweets
nohup python train_model.py --model w2v > out.txt &
```
	
### Test 
* Input file format - **note that the text should be tokenized**

| tokenized tweet        |
|------------------------|
| I may be the only one… |
| If Scotland woke…      |

	
	

```bash
cd cnntweets
nohup python test_model.py --model w2v
```
