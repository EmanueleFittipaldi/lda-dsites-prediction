import pprint
import nltk
import re
import gensim
from gensim import models
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation,strip_numeric
from gensim import corpora
from textblob import Word
from textblob import TextBlob
from collections import defaultdict
from bs4 import BeautifulSoup


#Con questo .py cerco di trovare un modo con cui riuscire ad esptrapolare le review da qualsiasi sito del darkweb.
#Alla luce di ciò, non è più possibile andare ad ancorarsi alla struttura specifica di un sito in particolare, per identiricare
#un pattern ricorrente nella disposizione degli elementi HTML, come si fa di solito durante il processo di scraping.
#L'approccio utilizzato è il seguente. Mediante questa funzione extractText al quale si passa un oggetto di tipo BeautifulSoup
#riusciamo ad estrarre tutto il testo, esente da tag html, presente nel sito. Questo è il massimo che si riesce a fare, adottando
#un approccio generico. Per cercare di filtrare il testo, estrapolando solo i commenti, ho fatto affidamento al preprocessing.
#In particolare, quello che è stato fatto è, prima di tutto prendere il documento generato dall'estrazione del testo presente in
#tutta la pagina, dividerlo in una lista di stringhe, dove ogni stringa corrisponde ad una riga del documento iniziale.

#DARKMARKET
#E:\ANITA-Dumps-Uncompressed\Sample 1\darkmarketsomqvzqfjudpd6t5eabgvvpplrbtzq6prervyogenlrlqd.onion\vendor\0c2c8cf0-423a-11ea-ba8c-61886ab94bd5\feedback.html

#BERLUSCONI_MARKET
#E:\ANITA-Dumps-Uncompressed\Sample 2\2019_08_20\2019_08_20_berlusconi_c=listings_a=vendor_v_id=2d9a6127f72fe3007ea30a1be8cbb104_tab=4.htm

#AGARTHA
#E:\ANITA-Dumps-Uncompressed\Sample 1\agartha_2020-03-31_18-36-01\agarthafw2cock27.onion\vendor\AmazingShop\ratings.html

#-------------------------------------------------------------------------------FUNCTIONS----------------------------------------------------------------------
def extractText(soup):
    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()
    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return text

def splitAndLower(corpus):
    splitList=[]
    splitList = txToList.split("\n")
    print("\nText splitted into a list of strings:")
    print(splitList)

    filteredCorpus = []
    for document in splitList:
        filteredCorpus.append(document.lower())
    print("\nLowerCasing:")
    print(filteredCorpus)

    return filteredCorpus

def TokenizingCorpus(corpus):
    new_corpus = []
    for document in corpus:
        new_corpus.append(gensim.utils.simple_preprocess(document, deacc=True, min_len=4, max_len=15))

    print("\nTokenized documents:")
    print(new_corpus)

    # rimuovo le liste vuote che non servono a niente ed interferiscono con i calcoli
    TextExtracted = [x for x in new_corpus if x != []]
    print("\nDeleting empty Lists:")
    print(TextExtracted)
    return TextExtracted

def delUnder4Words(corpus):
    prettyList = []
    for document in corpus:
        if len(document) > 3:
            prettyList.append(document)
    print("\nReady for refinement:")
    print(prettyList)
    return prettyList

def noPuncNoNumb(corpus):
    List_No_punct_numb = [[strip_punctuation(stringa) and strip_numeric(stringa) for stringa in group] for group in
                          corpus]

    print("\nList_No_punct_numb:")
    print(List_No_punct_numb)
    return List_No_punct_numb

def errataCorrige(corpus):
    List_spellChecked = [[str(TextBlob(text).correct()) for text in document]for document in corpus]
    return List_spellChecked

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(contractions.keys()))
def expandContractions(text, c_re=c_re):
    def replace(match):
        return contractions[match.group(0)]
    return c_re.sub(replace, text)

def deContract(corpus):
    listExpanded = [[expandContractions(text) for text in document]for document in corpus]
    print("listExpanded")
    print(listExpanded)
    return listExpanded

def singAndLemm(corpus):
    newList = [[ (TextBlob(text)).words[0].singularize() for text in document]for document in corpus]
    print("\nSingularized lists:")
    print(newList)
    return lemmatize(newList)

# nota: la lemmatizzazione di textBlob richiede di indicare se deve lemmatizzare un verbo o un sostantivo.
# dato che non ho previsto una fase di POS, mi accontento (per ora ) di lemmatizzare soltanto i verbi
# in quanto sembrano essere quelli che hanno più necessità e che racchiudono il significato maggiore
def lemmatize(corpus):
    newList = [[text.lemmatize("v") for text in document] for document in corpus]
    print("\nLemmatized lists:")
    print(newList)
    return newList

def reStopWords(corpus):
    newList=[[remove_stopwords(text) for text in document]for document in corpus]
    newList=[[x for x in document if x != ''] for document in newList]
    print("\nStopwords removed:")
    print(newList)
    return newList
#-----------------------------------------------------Step 0 - Opening the file and extracting the HTML as a BeautifulSoup Object--------------------------------
file=open(r"E:\ANITA-Dumps-Uncompressed\Sample 1\agartha_2020-03-31_18-36-01\agarthafw2cock27.onion\vendor\AmazingShop\ratings.html","r")
soup = BeautifulSoup(file, features="html.parser")


#-----------------------------------------------------Step 1 - Getting the text from an HTML file---------------------------------------------------------------
txToList= extractText(soup)
print("Testo estratto:")
print(txToList)
print("\n")


#-----------------------------------------------------Step 2 - Splitting the document into a List of strings----------------------------------------------------
# Here I am splitting the whole string extracted from the website into little chunks. I am splitting It into several lines, everytime a "\n" is encountered.
# Then I lower each line. From this step, we obtain the raw string in a workable format, from which we can apply further processing.
filteredCorpus=splitAndLower(txToList)


#----------------------------------------Step 3 - Tokenizing every string and removing tokens shorter than 4 characters and longer than 15 character------------
# Here I remove all the words of length <4 or >15. This will empty some lists, hence I need to remove them.
TextExtracted= TokenizingCorpus(filteredCorpus)


#----------------------------------------Step 4 - Creating a new List, keeping only documents with at least 4 words-------------------------------------------
# Here I apply the observation that all the lines that are less than 4 words, have a high probability of being just trash lines, that I need to remove in order
# to extract the text from the reviews. So, here, I am creating another list where I save only the lists which have a length >=4
prettyList=delUnder4Words(TextExtracted)


#----------------------------------------Step 5  -  Eliminare la punteggiatura e numeri----------------------------------------------------------------
List_No_punct_numb= noPuncNoNumb(prettyList)


#----------------------------------------Step 6  -  Spelling correction--------------------------------------------------------------------------------
List_spellChecked=errataCorrige(List_No_punct_numb)
print("\n")
print("List_spellChecked")
print(List_spellChecked)
print("\n")


#----------------------------------------Step 7  -  Espansione delle contrazioni-----------------------------------------------------------------------
List_deContracted =deContract(List_spellChecked)


#----------------------------------------Step 8  -  Singolarizzazione e lemmatizzazione----------------------------------------------------------------------------------
List_singAndLem = singAndLemm(List_deContracted)


#----------------------------------------Step   -  StopWords removal-------------------------------------------------------------------------------------------------
List_final = reStopWords(List_singAndLem)


#in questo dizionario sono presenti le parole con assegnata la frequenza
frequency = defaultdict(int)
for text in List_final:
    for token in text:
        frequency[token] += 1
print("\n")
print(frequency)

# manteniamo soltanto le parole che appaiono più di una volta e meno di 30 volta
processed_corpus = [[token for token in text if frequency[token] > 1 and frequency[token]<30] for text in List_final]
pprint.pprint(processed_corpus)

#ricalcoliamo la frequenza, dato che adesso abbiamo eliminato delle parole, giusto per avere il grafico aggiornato
frequency = defaultdict(int)
for text in processed_corpus:
    for token in text:
        frequency[token] += 1
print("\n")
print(frequency)

#Per fare il plotting del dizionario, per rappresentare le parole/frequenza sugli assi
import matplotlib.pylab as plt
lists = sorted(frequency.items()) # sorted by key, return a list of tuples
x, y = zip(*lists) # unpack a list of pairs into two tuples
plt.plot(x, y)
plt.show()


# processed_corpus= List_final


#Creiamo un dizionario in cui ci sono tutti i termini a cui associamo un ID unico
from gensim import corpora
dictionary = corpora.Dictionary(processed_corpus)
print(dictionary)

print("\nid di ogni parola contenuta nel dizionario")
print(dictionary.token2id)

print("\nconvertiamo ogni documento in una rappresentazione bag of words, ovvero dei vettori\n")
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
print(bow_corpus)

# Alleniamo il modello sul nostro bow_corpus
tfidf = models.TfidfModel(bow_corpus)

#salviamo il modello su disco in modo da poter riprendere il training in seguito
# tfidf.save("C:\\Users\\emanu\\Desktop\\Appunti- on Drive\Tirocinio\\modello.tfidf_model")

#applico il modello tfidf alle parole
corpusInTfidf=[]
for x in processed_corpus:
    corpusInTfidf.append(tfidf[dictionary.doc2bow(x)])

print(*corpusInTfidf,sep="\n")

