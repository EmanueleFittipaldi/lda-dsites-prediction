import gensim
import pickle
import numpy as np
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
from geneticalgorithm import geneticalgorithm as ga

# Apro il Corpora serializzato nella nella fase di pre-processing del testo affrontata eseguendo lo script
# Python "preProcessing_Pipeling.py".
with open ('Serialized_Processed_Corpora', 'rb') as fp:
    LISTA_CORPUS_PREPROCESSATI = pickle.load(fp)

#dato che LISTA_CORPUS_PREPROCESSATI potrebbe contenere qualche lista vuota, mi assicuro di eliminarle preventivamente
Corpora = [x for x in LISTA_CORPUS_PREPROCESSATI if x]
print("Corpora su cui applicherò il Bag of words")
print(Corpora)
print("\n")

#Creo un dizionario dal corpora, il quale contiene la frequenza di ogni parola.
dictionary= gensim.corpora.Dictionary(Corpora)
count = 0
for k, v in dictionary.iteritems():
        #print(k,v)
        count +=1

dictionary.filter_extremes(no_below=15,no_above=0.5,keep_n=600)
bow_corpus = [dictionary.doc2bow(doc) for doc in Corpora]

def CalculateCoherence(X):
    x = int(X[0])
    y = int(X[1])
    print("LDA parameters:\npasses:"+str(x)+"\nnum_topics:"+str(y)+"\n")
    # Alleno il modello LDA
    lda = LdaModel(bow_corpus, id2word=dictionary, passes=x, num_topics=y)
    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda, texts=Corpora, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score: ', coherence_lda)
    return coherence_lda

# Passes da 1 a 99, num_topics da 2 a 99
varbound=np.array([[1,100],[2,100]])
algorithm_param = {'max_num_iteration': 5,\
                   'population_size':5,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':3}

Genetic_Algorithm=ga(function=CalculateCoherence,dimension=2,variable_type='int',variable_boundaries=varbound,algorithm_parameters=algorithm_param)
print("\nSettaggi scelti per l'algoritmo genetico")
print(Genetic_Algorithm.param)
Genetic_Algorithm.run()
output_dict={}
convergence=Genetic_Algorithm.report
solution=Genetic_Algorithm.ouput_dict
print("\nSoluzione:")
print(solution)

""" Settaggi scelti per l'algoritmo genetico
{'max_num_iteration': 5, 'population_size': 5, 'mutation_probability': 0.1, 'elit_ratio': 0.01, 'crossover_probability': 0.5, 'parents_portion': 0.3, 'crossover_type': 'uniform', 'max_iteration_without_improv': 3}
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.484003954674659
LDA parameters:
passes:7
num_topics:88

Coherence Score:  -7.423403241409457
LDA parameters:
passes:7
num_topics:34

Coherence Score:  -7.639967055687163
LDA parameters:
passes:50
num_topics:32

Coherence Score:  -8.149792159810666
LDA parameters:
passes:44
num_topics:89

Coherence Score:  -7.232784068825673
||||||||||________________________________________ 20.0% GA is running...LDA parameters:
passes:92
num_topics:33

Coherence Score:  -7.749466667814764
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.231932726015275
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.080653945008647
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -7.617599181522642
||||||||||||||||||||______________________________ 40.0% GA is running...LDA parameters:
passes:8
num_topics:33

Coherence Score:  -7.645945787139154
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.06578576518604
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -7.919563258333094
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.61344171711911
||||||||||||||||||||||||||||||____________________ 60.0% GA is running...LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.140733118322242
LDA parameters:
passes:92
num_topics:28

Coherence Score:  -9.073063212742586
LDA parameters:
passes:92
num_topics:33

Coherence Score:  -8.447181838828412
LDA parameters:
passes:92
num_topics:4

Coherence Score:  -8.930667773658655
||||||||||||||||||||||||||||||||||||||||__________ 80.0% GA is running...LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.685976079807688
LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.231397626751512
LDA parameters:
passes:92
num_topics:6

Coherence Score:  -8.421722159145245
LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.72753805567032
|||||||||||||||||||||||||||||||||||||||||||||||||| 100.0% GA is running...LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.430941578640583
LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.634669489057737
LDA parameters:
passes:92
num_topics:28

Coherence Score:  -8.510523042405094
LDA parameters:
passes:92
num_topics:4

Coherence Score:  -7.016803827764668
 The best solution found:
 [92. 28.]

 Objective function:
 -9.073063212742586
"""