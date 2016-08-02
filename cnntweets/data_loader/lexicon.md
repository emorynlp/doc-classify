## Lexicon dataset


## Source
* From [this](http://saifmohammad.com/WebPages/lexicons.html)


## File format

### HashtagSentimentAffLexNegLex
* filename: HS-AFFLEX-NEGLEX-unigrams.txt
* f = 3 (3 dim vector)
	* dim=1: <score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
	* dim=2: <Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
	* dim=3: <Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.

* N = 43,949

### MaxDiff-Twitter-Lexicon
* filename: Maxdiff-Twitter-Lexicon_0to1.txt
* f = 1 (scalar)
	* range: [0 1], 0 (most negative) and 1 (most positive)
	* **neutral is 0.5**
* N = 1,515


### NRC-Hashtag-Sentiment-Lexicon-v0.1
* filename: unigrams-pmilexicon.txt
* f = 3
	* dim = 1: sentimentScore is a real number. A positive score indicates positive sentiment. A negative score indicates negative sentiment.
	* dim = 2: numPositive is the number of times the term co-occurred with a positive marker such as a positive emoticon or a positive hashtag
	* dim = 3: numNegative is the number of times the term co-occurred with a negative marker such as a negative emoticon or a negative hashtag.
* N = 54,129

### Sentiment140-Lexicon-v0.1
* filename: unigrams-pmilexicon_sentiment_140.txt
* f = 3
	* dim = 1: Terms with a non-zero PMI score with positive emoticons and PMI score of 0 with negative emoticons were assigned a sentimentScore of 5.
	  Terms with a non-zero PMI score with negative emoticons and PMI score of 0 
	  with positive emoticons were assigned a sentimentScore of -5.

	* dim = 2: numPositive is the number of times the term co-occurred with a positive marker such as a positive emoticon or a positive emoticons.

	* dim = 3: numNegative is the number of times the term co-occurred with a negative marker such as a negative emoticon or a negative emoticons.
* N = 62,468

### Sentiment140AffLexNegLex
* filename: S140-AFFLEX-NEGLEX-unigrams.txt
* f = 3
	* dim = 1: <score> is a real-valued sentiment score: score = PMI(w, pos) - PMI(w, neg), where PMI stands for Point-wise Mutual Information between a term w and the positive/negative class;
	* dim = 2: <Npos> is the number of times the term appears in the positive class, ie. in tweets with positive hashtag or emoticon;
	* dim = 3: <Nneg> is the number of times the term appears in the negative class, ie. in tweets with negative hashtag or emoticon.

* N = 55,146

### ??
* filename: EverythingUnigramsPMIHS.txt
* f = 1
* N = 4,376



