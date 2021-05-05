import pprint
import nltk
import re
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
from csv import reader
import pandas as pd


# Importing all the functions needed for the preprocessing...
from functions import Lower
from functions import TokenizingCorpus
from functions import delUnder4Words
from functions import noPuncNoNumb
from functions import errataCorrige
from functions import expandContractions
from functions import deContract
from functions import singAndLemm
from functions import lemmatize
from functions import reStopWords

""" Per validare la bontà del modello devo andare a suddividere il dataset in due porzioni. Una porzione dedicata al training del modello
    ed una porzione dedicata al testing del modello. Possiamo pensare di dividere il Dataset in 80% Training e 20% Testing. Nel caso volessimo
    utilizzare la k fold cross validation dovremmo procedere diversamente. Dovrei partizionare il dataset in k partizioni, allenare ogni volta
    il modello sulle k-1 partizioni e usare la k-esima partizione come partizione di validazioni, e ripetere il procedimento per tutte le k
    partizioni. Per limitazioni dovute alla macchina su cui sto lavorando questo non è una strategia di sviluppo rapida.
    
    Per riuscire a validare il modello devo creare una corrispondenza tra le recensioni, I topic in cui esse ricadono e quale era la loro label di partenza.
    Quindi devo creare prima un dataset con le sole recensioni in modo da poter arrivare a costruire ed allenare LDA. Sfrutto poi il Dataset iniziale con le
    label, do in pasto ad LDA l'i-esimo documento con i che va da 1 a k, vedo a quale Topic viene classificato e creo un nuovo Dataset in cui annoto che il 
    documento i-esimo è ricaduto nel Topic x ed ha label y. Così facendo ogni qual volta che un determinato documento ricade in un certo topic so che quel topic
    ha una certa probabilità di essere legato alla label y """



#----------------------------------------------------- Step 0 I read the CSV -----

data=pd.read_csv("reviews.csv")
# Remove the columns
print("\nI remove the unwanted columns 'Unnamed: 0','id', 'package_name', 'date','star', 'version_id'")
data = data.drop(columns=['Unnamed: 0','id', 'package_name', 'date',
                              'star', 'version_id'], axis=1)
#mi prendo un attimo tutte le recensioni e le metto in una lista
temp_data = data
temp_data = temp_data.drop(columns=['category'],axis=1)
r=temp_data.reset_index().values.tolist()
print(r)
reviews=[]
for lista in r:
    temp_list=[lista[1]]
    reviews.append(temp_list)
#print(reviews)  #questo costituisce il testo da lavorare

print("\nQueste sono le colonne estratte:")
print(data.columns)
print("\n Questi sono i primi 10 elementi del CSV")
print(data.head(10))

print("\nAggiungo la colonna che conterrà la distribuzione dei topic associata ad ogni recensione labellata")
data['Topic_Distribuition']=""
print(data.head(5))

# print("lowercasing del testo")
#lowercasing all the text
# filteredCorpus = Lower(reviews)
# print(filteredCorpus[:10])

# ----------------------------------------Step 3 - Tokenizing every string and removing tokens shorter than 4 characters and longer than 15 character------------
# Here I remove all the words of length <4 or >15. This will empty some lists, hence I need to remove them.
# print("\nTokenizzazione del testo")
# TextExtracted = TokenizingCorpus(filteredCorpus)
# print(TextExtracted[:10])



# ----------------------------------------Step 4 - Creating a new List, keeping only documents with at least 4 words-------------------------------------------
# Here I apply the observation that all the lines that are less than 4 words, have a high probability of being just trash lines, that I need to remove in order
# to extract the text from the reviews. So, here, I am creating another list where I save only the lists which have a length >=4
# print("\nRimozione delle parole più corte di 5 caratteri")
# prettyList = delUnder4Words(TextExtracted)
# print(prettyList[:10])

# da problemi
# ----------------------------------------Step 5  -  Eliminare la punteggiatura e numeri----------------------------------------------------------------
# print("\nList_No_punct_numb:")
# List_No_punct_numb = noPuncNoNumb(TextExtracted)
# print(List_No_punct_numb[:10])

# ----------------------------------------Step 6  -  Spelling correction--------------------------------------------------------------------------------
# print("\nList_spellChecked:")
# List_spellChecked = errataCorrige(prettyList)
# print(List_spellChecked[:10])

# ----------------------------------------Step 7  -  Espansione delle contrazioni-----------------------------------------------------------------------
# print("espando le contrazioni")
# List_deContracted = deContract(prettyList)
#
# ----------------------------------------Step 8  -  Singolarizzazione e lemmatizzazione----------------------------------------------------------------------------------
#List_singAndLem = singAndLemm(List_deContracted)

# ----------------------------------------Step   -  StopWords removal-------------------------------------------------------------------------------------------------
#List_final = reStopWords(List_singAndLem)

# #-----------------------NON SO SE SERVE--------------#
#     # in questo dizionario sono presenti le parole con assegnata la frequenza
#     frequency = defaultdict(int)
#     for text in List_final:
#         for token in text:
#             frequency[token] += 1
#     # print("\n")
#     #print(frequency)
#
#     # manteniamo soltanto le parole che appaiono più di una volta e meno di 30 volta
#     processed_corpus = [[token for token in text if frequency[token] > 1 and frequency[token] < 30] for text in
#                         List_final]
#     #pprint.pprint(processed_corpus)
#
#
#     # Aggiorno le frequenze in questo dizionario sono presenti le parole con assegnata la frequenza
#     frequency = defaultdict(int)
#     for text in processed_corpus:
#         for token in text:
#             frequency[token] += 1
#     # print("\n")
#     # print(frequency)
# # -----------------------NON SO SE SERVE--------------#

#     #aggiungo il corpus processato alla lista che poi andrò ad utilizzare alla fine
#     #per generare il Bag of words
#     print("documento processato numero "+str(i))
#     i+=1
#     print(processed_corpus)
#
#     for sentence in processed_corpus:
#         LISTA_CORPUS_PROCESSATI.append(sentence)
#
# print("\nLISTA_CORPUS_PROCESSATI")
# print(LISTA_CORPUS_PROCESSATI)

#================================================================================================================================
# Per salvare il lavoro di preprocessing sul disco per non doverlo ripetere
# with open('Serialized_Processed_Corpora', 'wb') as fp:
#     pickle.dump(prettyList, fp)


# SNIPPET PER APRIRE IL CORPORA SERIALIZZATO
# Apro quanto serializzato
with open ('Serialized_Processed_Corpora', 'rb') as fp:
    LISTA_CORPUS_PREPROCESSATI = pickle.load(fp)

#Creo un dizionario dal corpora, contenente il numero delle volte che una
#parola appare nel training set
dictionary= gensim.corpora.Dictionary(LISTA_CORPUS_PREPROCESSATI)
count = 0
for k, v in dictionary.iteritems():
        print(k,v)
        count +=1

#filtriamo i token che appaiono in meno di 15 documenti (numero assoluto) o
#in più di 0.5 documenti (frazione della taglia totale del corpora, non il numero
#assoluto). Dopo i primi due step, manteniamo soltanto i 600 token più frequenti
#600 è un parametro arbitrario, ho scelto di provare per questo per iniziare dato
#che sono state trovate circa 700 parole diverse e vorrei cercare di mantenerne
#il più possibile
dictionary.filter_extremes(no_below=15,no_above=0.5,keep_n=600)

#Per ogni documento creiamo un dizionario che riporta quante parole e
#quante volte queste appaiono. Salviamo questo nell'oggetto bow_corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in LISTA_CORPUS_PREPROCESSATI]


# Train the model on the corpus.
lda = LdaModel(bow_corpus,id2word=dictionary,passes=20, num_topics=10)


#mostro a schermo tutti i topic trovati con la relativa distribuzione
for idx, topic in lda.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

#Per prendere la distribuzione di topic per ogni documento
print("\n\nDistribuzione dei topic sul documento 10")
top_topics = (lda.get_document_topics(bow_corpus[10], minimum_probability=0.0))
print(top_topics)



