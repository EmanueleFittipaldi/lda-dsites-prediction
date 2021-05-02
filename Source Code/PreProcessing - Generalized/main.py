import pprint
import nltk
import os
import gensim
import pickle
from gensim import models
from gensim.models import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation,strip_numeric
from gensim.test.utils import datapath
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim import corpora
from textblob import Word
from textblob import TextBlob
from collections import defaultdict
from bs4 import BeautifulSoup
from gensim.models import CoherenceModel
import numpy as np
from geneticalgorithm import geneticalgorithm as ga

#IMPORTING THE FUNCTIONS NEEDED TO PRE-PROCESS THE TEXT
from functions import getRatingsPaths
from functions import extractText
from functions import splitAndLower
from functions import TokenizingCorpus
from functions import delUnder4Words
from functions import noPuncNoNumb
from functions import errataCorrige
from functions import expandContractions
from functions import deContract
from functions import singAndLemm
from functions import lemmatize
from functions import reStopWords


#------------------------------------------ FASE 0 - APERTURA DEL CORPORA PREPROCESSATO E SERIALIZZATO -----------------------------------

# Apro il Corpora serializzato nella nella fase di pre-processing del testo affrontata eseguendo lo script
# Python "preProcessing_Pipeling.py".
with open ('Serialized_Processed_Corpora', 'rb') as fp:
    LISTA_CORPUS_PREPROCESSATI = pickle.load(fp)

#dato che LISTA_CORPUS_PREPROCESSATI potrebbe contenere qualche lista vuota, mi assicuro di eliminarle preventivamente
Corpora = [x for x in LISTA_CORPUS_PREPROCESSATI if x]
print("Corpora su cui applicherò il Bag of words")
print(Corpora)
print("\n")


#------------------------------------------ FASE 1 - CREAZIONE DEL DIZIONARIO ------------------------------------------------------------

#Creo un dizionario dal corpora, il quale contiene la frequenza di ogni parola.
dictionary= gensim.corpora.Dictionary(Corpora)
count = 0
for k, v in dictionary.iteritems():
        #print(k,v)
        count +=1

#filtriamo i token che appaiono in meno di 15 documenti (numero assoluto) o
#in più del 50% dei documenti. Dopo i primi due step, manteniamo soltanto i 600 token più frequenti
#600 è un parametro arbitrario, ho scelto di provare per questo per iniziare dato
#che sono state trovate circa 700 parole diverse e vorrei cercare di mantenerne
#il più possibile
dictionary.filter_extremes(no_below=15,no_above=0.5,keep_n=600)


#------------------------------------------ FASE 2 - RAPPRESENTAZIONE BoW ---------------------------------------------------------------

# Converto ogni documento nella sua rappresentazione Bag of Words sfruttando il dizionario creato in precedenza
bow_corpus = [dictionary.doc2bow(doc) for doc in Corpora]


#------------------------------------------ FASE 3 - TRAINING DI LDA  -------------------------------------------------------------------

# Alleno il modello LDA
lda = LdaModel(bow_corpus,id2word=dictionary,passes=92, num_topics=28)

# Printo i topic con le rispettive parole più significative appartenenti per ogni topic
for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


#------------------------------------------ FASE X - ESEMPIO DI CLASSIFICAZIONE  ----------------------------------------------------------

# Faccio vedere un esempio dove prendo un documento e tramite LDA vedo in quale topic ricade
# In base allo Score più alto che ottengo.
print("\ndocuemnto che sto cercando di classificare")
print(Corpora[1])
print("\n")

for index, score in sorted(lda[bow_corpus[1]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda.print_topic(index, 10)))















