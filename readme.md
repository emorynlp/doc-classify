## Emory Sentiment Analysis

* Author: [Bonggun Shin](bonggun.shin@emory.edu)

## Installation
### clone the repository

```bash
git clone git@github.com:emorynlp/doc-classify.git
```
* make virtual env

```bash
mkvirtualenv sent
```


### Dependencies

```bash
pip install -r requirements.txt
# install tensorflow depending on the server type (version should be 0.10.0rc0)
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl
pip install --upgrade $TF_BINARY_URL
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


## Usage
### Train 
* Input file format - **note that the text should be tokenized**

| line number | sentiment | tokenized tweet        |
|-------------|-----------|------------------------|
| 1           | object    | I may be the only one… |
| 2           | negative  | If Scotland woke…      |
	
```bash
cd cnntweets
# plain cnn
python train_model.py -v ./w2v-50.bin -t ./trn -d ./dev -l ./lex_config.txt -m ./mymodel
```
	
### Test 
* Input file format - **note that the text should be tokenized**

| tokenized tweet        |
|------------------------|
| I may be the only one… |
| If Scotland woke…      |


```bash
cd cnntweets
python decode.py -m ./mymodel -v ./w2v-50.bin  -l ./lex_config.txt -i ./input
```

