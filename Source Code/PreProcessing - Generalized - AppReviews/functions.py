import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_punctuation,strip_numeric
from textblob import TextBlob

def Lower(corpus):
    filteredCorpus = []
    for lista in corpus:
        temp_list=[]
        for parola in lista:
            temp_list.append(parola.lower())
        filteredCorpus.append(temp_list)
    # print("\nLowerCasing:")
    # print(filteredCorpus)

    return filteredCorpus

def TokenizingCorpus(corpus):
    new_corpus = []
    for lista in corpus:
        for parola in lista:
            new_corpus.append(gensim.utils.simple_preprocess(parola, deacc=True, min_len=4, max_len=15))

    # print("\nTokenized documents:")
    # print(new_corpus)

    # rimuovo le liste vuote che non servono a niente ed interferiscono con i calcoli
    TextExtracted = [x for x in new_corpus if x != []]
    # print("\nDeleting empty Lists:")
    # print(TextExtracted)
    return TextExtracted

def delUnder4Words(corpus):
    prettyList = []
    for lista in corpus:
        temp_list=[]
        for document in lista:
            if len(document) > 4:
                temp_list.append(document)
        prettyList.append(temp_list)

    # print("\nReady for refinement:")
    # print(prettyList)
    return prettyList

def noPuncNoNumb(corpora):
    List_No_punct_numb = [[[strip_punctuation(stringa) and strip_numeric(stringa) for stringa in group] for group in
                          corpus] for corpus in corpora]

    # print("\nList_No_punct_numb:")
    # print(List_No_punct_numb)
    return List_No_punct_numb

def errataCorrige(corpora):
    risultato=[]
    for lista in corpora:
        for parola in lista:
            temp_lista=[]
            temp_lista.append(str(TextBlob(parola).correct()))
            risultato.append(temp_lista)
    return risultato
    # List_spellChecked = [[str(TextBlob(text).correct()) for text in document]for document in corpora]
    # return List_spellChecked

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
    # print("listExpanded")
    # print(listExpanded)
    return listExpanded

def singAndLemm(corpus):
    newList = [[ (TextBlob(text)).words[0].singularize() for text in document]for document in corpus]
    # print("\nSingularized lists:")
    # print(newList)
    return lemmatize(newList)

# nota: la lemmatizzazione di textBlob richiede di indicare se deve lemmatizzare un verbo o un sostantivo.
# dato che non ho previsto una fase di POS, mi accontento (per ora ) di lemmatizzare soltanto i verbi
# in quanto sembrano essere quelli che hanno più necessità e che racchiudono il significato maggiore
def lemmatize(corpus):
    newList = [[text.lemmatize("v") for text in document] for document in corpus]
    # print("\nLemmatized lists:")
    # print(newList)
    return newList

def reStopWords(corpus):
    newList=[[remove_stopwords(text) for text in document]for document in corpus]
    newList=[[x for x in document if x != ''] for document in newList]
    # print("\nStopwords removed:")
    # print(newList)
    return newList