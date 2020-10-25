pip install num2words


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from os.path import isfile
from os.path import join

import os
from num2words import num2words
import numpy as np
import string
import pandas as pd
import math
import time


def lowercase(data):
    """changes the case of all characters in the document to lowercase"""
    return np.char.lower(data)


def remove_stopwords(data):
    """removes stopwords from the document"""
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new = ""
    for word in words:
        if word not in stop_words and len(word) > 1:
            new = new + " " + word
    return new


def remove_punct(data):
    """removes all punctuation from the document"""
    punct = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(punct)):
        data = np.char.replace(data, punct[i], ' ')
        data = np.char.replace(data, " ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophes(data):
    """removing apostrophes separately"""
    data = np.char.replace(data, "'", "")
    data = np.char.replace(data, "â\x80\x98", "") #removing unicode apostrophes
    data = np.char.replace(data, "â\x80\x99", "")
    return data


def stemming(data):
    """performing stemming on the tokens in the document"""
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new = ""
    for word in tokens:
        new = new + " " + stemmer.stem(word)
    return new


def lemmatize(data):
    """lemmatizing the document"""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new = ""
    for word in tokens:
        new = new + " " + lemmatizer.lemmatize(word)
    return new


def num_to_words(data):
    """converting nunmbers to words in the document"""
    tokens = word_tokenize(str(data))
    new = ""
    for word in tokens:
        try:
            word = num2words(int(w))
        except:
            a = 0
        new = new + " " + word
    new = np.char.replace(new, "-", " ")
    return new


def normalize(data):
    """combining all the above functions in a suitable order"""
    data = lowercase(data)
    data = remove_punct(data)
    data = remove_apostrophes(data)
    data = remove_stopwords(data)
    data = num_to_words(data)
    data = lemmatize(data)
    data = stemming(data)
    data = remove_punct(data)
    data = num_to_words(data)
    data = lemmatize(data)
    data = stemming(data)
    data = remove_punct(data) #done again to remove hyphens produced by num2words
    data = remove_stopwords(data) #done agan to remove stopwords produced by num2words
    return data


#computing tf dictionary

def calcTFdict(doc):
    """Returns a term frequency dictionary for each document, 
    with keys that are unique tokens in the document and values are the corresponding term frequencies"""
    
    TFDict = {}
    
    #counts number of appearances of term in document
    for term in doc:
        if term in TFDict.keys():
            TFDict[term] +=1
        else:
            TFDict[term] = 1
            
    #Computing tf for each term
    for key in TFDict:
        TFDict[key] = TFDict[key]/len(doc)
    
    return TFDict


def calcCountDict(TFdict):
    """Returns dictionary with keys as all the unique terms in corpus and values 
    is the number of documents in which each term appears"""
    
    countDict = {}
    
    for doc in TFdict:
        for term in doc:
            if term in countDict:
                countDict[term] +=1
            else:
                countDict[term] = 1
                
    return countDict


#computing idf dictionary

def calcIDFDict(countDict, numfiles):
    """Returns dictionary whose keys are all unique words in dataset and 
    values are corresponding Inverse Document Frequencies"""
    
    IDFDict = {}
    for term in countDict:
        IDFDict[term] = math.log(numfiles / countDict[term])
    
    return IDFDict


#calculating TF-IDF dictionary
def calcTFIDFDict(TFDict, IDFDict):
    """Returns dictionary whose keys are all unique terms in the document 
    and values are corresponding TF-IDF value"""

    TFIDFDict = {}
    
    #for each term in the document, multiply the tf and idf values
    
    for term in TFDict:
        TFIDFDict[term] = TFDict[term] * IDFDict[term]

    return TFIDFDict


def calc_TF_IDF_Vector(doc, termDict):
    """Creating TF-IDF vector (for calculating cosine similarity)"""
    TFIDFVec = [0.0] * len(termDict)
    
    #for each unique term, if it is in the document, store the TF-IDF value
    for i, term in enumerate(termDict):
        if term in doc:
            TFIDFVec[i] = doc[term]
        
    return TFIDFVec


def dot_product(a, b):
    """returns dot product of two vectors"""
    dp = 0.0
    for i, j in zip(a, b):
        dp += i * j
    return dp


def norm(vec):
    """returns the norm or magnitude of a vector"""
    n = 0.0
    for i in vec:
        n += math.pow(i, 2)
    return math.sqrt(n)


def cosine_similarity(a, b):
    """returns cosine similarity score of two vectors"""
    cs = dot_product(a, b)/(norm(a) * norm(b))
    return cs


# ALL THE CELLS AFTER THIS POINT SHOULD BE RUN EVERY TIME THE TEST OR TRAINING SETS ARE MODIFIED. THE ABOVE CELLS NEED NOT BE RE-RUN


start_time = time.time()
normalized_trg = []
normalized_test = []
path_trg = "./texts/" #directory in which training set is located
path_test = "./test/"
#test_file = input("Enter file name: ") #g4pC_taska.txt
trg_files = [document for document in os.listdir(path_trg) if document.endswith('.txt')]
test_files = [document for document in os.listdir(path_test) if document.endswith('.txt')]


numfiles_trg = 0 #number of files in the training directory
for file in trg_files:
    file.encode('utf8').strip() #encodes each of the files into utf-8
    fh = open(os.path.join(path_trg, file), 'r', encoding = "utf-8")
    file_content = fh.read()
    numfiles_trg = numfiles_trg + 1

    normalized_trg.append(word_tokenize(str(normalize(file_content)))) #performing normalization on the training files


numfiles_test = 0 #number of files in the testing directory
for file in test_files:
    file.encode('utf8') #encodes each of the files into utf-8
    fh = open(os.path.join(path_test, file), 'r', encoding = "utf-8")
    file_content = fh.read()
    numfiles_test = numfiles_test + 1

    normalized_test.append(word_tokenize(str(normalize(file_content)))) #performing normalization on the test files


normalize_time = time.time() - start_time


print("Normalizing time:", normalize_time, "seconds")


exec_start_time = time.time()


"""adding test file to the total corpus so that we can perform TF-IDF vectorization"""
normalized_corpus = normalized_trg + normalized_test 
test_doc_index_start = len(normalized_corpus) - numfiles_test
numfiles = numfiles_trg + numfiles_test


TFdict = [] #term frequency dictionary of the corpus
for i in range(len(normalized_corpus)):
    d = calcTFdict(normalized_corpus[i])
    TFdict.append(d)


countDict = calcCountDict(TFdict)
#calculating the number of documents in which each term appears


IDFDict = calcIDFDict(countDict, numfiles)
#calculating the IDF dictionary of the corpus


TFIDFDict = [calcTFIDFDict(doc, IDFDict) for doc in TFdict]
#calculating the TF-IDF dictionary


termDict = sorted(countDict.keys())


tf_idf_vector = [calc_TF_IDF_Vector(doc, termDict) for doc in TFIDFDict]
#vectorizing the TF-IDF dictionary for the corpus


similarity_scores = {new_list: [] for new_list in range(numfiles_test)}
"""calculating the cosine similarity of each of the documents in the training set with respect to the test document"""
for i in range(len(tf_idf_vector) - numfiles_test):
    for j in range(numfiles_test):
        cs = cosine_similarity(tf_idf_vector[(test_doc_index_start + j)], tf_idf_vector[i])
        similarity_scores[j].append(cs) 


# THE NEXT FEW CELLS ARE FOR PRINTING THE RANKED LIST FOR EACH OF THE TEST FILES

ranked_dict_keys = trg_files
ranked_dict = {new_list: [] for new_list in range(len(ranked_dict_keys))}


for i in range(len(similarity_scores)):
    ranked_dict_values = similarity_scores[i]
    ranked_dict[i] = {ranked_dict_keys[i]: ranked_dict_values[i] for i in range(len(ranked_dict_keys))}


ranked_dict_view = {new_list: [] for new_list in range(len(similarity_scores))}
for i in range(len(ranked_dict)):
    if(ranked_dict[i]):
        ranked_dict_view[i] = [ (v,k) for k,v in ranked_dict[i].items() ]
        ranked_dict_view[i].sort(reverse = True) # natively sort tuples by first element
        
        print("\n", test_files[i], "similarity ranking:\n")
        for v,k in ranked_dict_view[i]:
            print(k, ":", str((v * 100)) + "%", "similarity")
    else:
        break

exec_time = time.time() - exec_start_time
print("Execution time: ", exec_time, "seconds")
