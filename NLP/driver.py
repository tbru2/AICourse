import os
import tarfile
from six.moves import urllib
import numpy as np
import csv
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer

train_path = "../resource/lib/publicdata/aclImdb/train/" # use terminal to ls files under this directory
test_path = "../resource/asnlib/public/imdb_te.csv" # test data for grade evaluation
data_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"



def imdb_data_preprocess(inpath=data_url, outpath="./", name="imdb_tr.csv", mix=False):
 	#'''Implement this module to extract
	#and combine text files under train_path directory into 
    #imdb_tr.csv. Each text file in train_path should be stored 
    #as a row in imdb_tr.csv. And imdb_tr.csv should have two 
    #columns, "text" and label'''
    dir = "./aclImdb/train/neg"
    cvsFile = open("imdb_tr.csv", 'w')
    dataWriter = csv.writer(cvsFile,delimiter=",")
    for filename in os.listdir(dir):
        fp = open(os.path.join(dir,filename),'r')
        for line in fp:
            dataWriter.writerow([line, "0"])
        fp.close()
    dir = "./aclImdb/train/pos"
    for filename in os.listdir(dir):
        fp = open(os.path.join(dir,filename),'r')
        for line in fp:
            dataWriter.writerow([line,'1'])
        fp.close()

def extract_test_set():
    dir = "./aclImdb/test/neg"
    cvsFile = open("imdb_tr_test.csv", 'w')
    dataWriter = csv.writer(cvsFile,delimiter=",")
    for filename in os.listdir(dir):
        fp = open(os.path.join(dir,filename),'r')
        for line in fp:
            dataWriter.writerow([re.sub(r'\W+',' ',line), "0"])
        fp.close()
    dir = "./aclImdb/test/pos"
    for filename in os.listdir(dir):
        fp = open(os.path.join(dir,filename),'r')
        for line in fp:
            dataWriter.writerow([re.sub(r'\W+',' ',line),'1'])
        fp.close()

def main():
    #imdb_data_preprocess()
    #extract_test_set()

    stopWordsFile = open("./stopwords.txt",'r')
    stopWords = [line.strip("\n") for line in stopWordsFile]
    stopWords.append('.')
    stopWordsFile.close()

    data = pd.read_csv('imdb_tr.csv', delimiter=',',encoding = "ISO-8859-1")
    test = pd.read_csv('imdb_te.csv', delimiter=',',encoding = "ISO-8859-1")
    df = pd.DataFrame(data)
    df_test = pd.DataFrame(test)
    X = df.values
    X_test = df_test.values
    np.random.shuffle(X)
    x_train, y_train = X[:,0], X[:,1]
    y_train = y_train.astype('int')

    x_test = X_test[:,1]
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1),stop_words=stopWords, lowercase=True,encoding = "ISO-8859-1")),
        ('clf', SGDClassifier(loss="hinge", penalty="l1"))
    ])

    clf = pipeline.fit(x_train, y_train)
    
    clf_pred = clf.predict(x_test)
    fp = open("unigram.output.txt", 'w')
    for pred in clf_pred:
        fp.write(str(pred) + '\n')
   
    fp.close()

    tfidf_pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1),stop_words=stopWords, encoding = "ISO-8859-1",lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss="hinge", penalty="l1"))
    ])
    clf = tfidf_pipeline.fit(x_train,y_train)
    clf_pred = clf.predict(x_test)
    fp = open("unigramtfidf.output.txt",'w')
    for pred in clf_pred:
        fp.write(str(pred) + '\n')
    fp.close()

    
    bigram_pipeline = Pipeline([
        ('vect',CountVectorizer(ngram_range=(1,2),stop_words=stopWords, max_features=6000, encoding = "ISO-8859-1")),
        ('clf', SGDClassifier(loss="hinge", penalty="l1"))
    ])
   
    clf = bigram_pipeline.fit(x_train, y_train)
    clf_pred = clf.predict(x_test)
    fp = open("bigram.output.txt", 'w')
    for pred in clf_pred:
        fp.write(str(pred) + '\n')
    fp.close()

    tBigram_pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,2),stop_words=stopWords, max_features=6000, encoding = "ISO-8859-1")),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier(loss="hinge", penalty="l1"))
    ])

    clf = tBigram_pipeline.fit(x_train, y_train)
    
    clf_pred = clf.predict(x_test)
    fp = open("bigramtfidf.output.txt",'w')
    for pred in clf_pred:
          fp.write(str(pred) + '\n')
    fp.close()
if __name__ == "__main__":
    main()
 	#'''train a SGD classifier using unigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #unigram.output.txt'''
  	
    #'''train a SGD classifier using bigram representation,
    #predict sentiments on imdb_te.csv, and write output to
    #unigram.output.txt'''
     
    # '''train a SGD classifier using unigram representation
    # with tf-idf, predict sentiments on imdb_te.csv, and write 
    # output to unigram.output.txt'''
  	
     #'''train a SGD classifier using bigram representation
     #with tf-idf, predict sentiments on imdb_te.csv, and write 
     #output to unigram.output.txt'''
 

