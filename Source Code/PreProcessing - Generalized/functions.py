import os
import re

def getRatingsPaths(DatasetFolder):
    # Prendo tutti i percorsi inerenti alle cartelle chiamate vendor
    Vendors_folders = []
    for root, subdirs, files in os.walk(DatasetFolder):
        for d in subdirs:
            if d == "vendor":
                Vendors_folders.append(os.path.join(root, d))
    # print("\nPercorsi delle cartelle contenenti i venditori\n")
    # pprint(Vendors_folders)

    # Prendo tutti i percorsi di tutti i vendor contenuti nelle cartelle vendor
    # Ogni cartella è un vendor, il cui nome della cartella corrisponde al nome del vendor
    Vendors_names = []
    for VendorPath in Vendors_folders:
        p = os.listdir(VendorPath)  # Mi lista tutto quello che c'è in nella cartella vendor
        for file in p:
            if os.path.isdir(os.path.join(VendorPath, file)):
                Vendors_names.append(os.path.join(VendorPath, file))

    # print("\nPercorsi delle cartelle di ogni venditore\n")
    # pprint(Vendors_names)

    # Creo tutti i percorsi finali che mi portano ai ratings.html, dove sono contenute le recensioni
    Final_Paths = []
    for Vendor in Vendors_names:
        if(((os.listdir(Vendor)).pop(0)).endswith(".html")):
            Final_Paths.append(os.path.join(Vendor, (os.listdir(Vendor)).pop(0))) #in alcune cartelle non c'è solo ratings, ma c'è prima una icona, poi ratings
        else:                                                                     #quindi se il primo elemento non è un html, prendo il secondo. (Migliorabile)
            Final_Paths.append(os.path.join(Vendor, (os.listdir(Vendor)).pop(1)))

    # print("\nPercorsi dei file delle recensioni per ogni venditore\n")
    # pprint(Final_Paths)
    return Final_Paths

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
    # print("\nText splitted into a list of strings:")
    # print(splitList)

    filteredCorpus = []
    for document in splitList:
        filteredCorpus.append(document.lower())
    # print("\nLowerCasing:")
    # print(filteredCorpus)

    return filteredCorpus

def TokenizingCorpus(corpus):
    new_corpus = []
    for document in corpus:
        new_corpus.append(gensim.utils.simple_preprocess(document, deacc=True, min_len=4, max_len=15))

    # print("\nTokenized documents:")
    # print(new_corpus)

    # rimuovo le liste vuote che non servono a niente ed interferiscono con i calcoli
    TextExtracted = [x for x in new_corpus if x != []]
    # print("\nDeleting empty Lists:")
    # print(TextExtracted)
    return TextExtracted

def delUnder4Words(corpus):
    prettyList = []
    for document in corpus:
        if len(document) > 3:
            prettyList.append(document)
    # print("\nReady for refinement:")
    # print(prettyList)
    return prettyList

def noPuncNoNumb(corpus):
    List_No_punct_numb = [[strip_punctuation(stringa) and strip_numeric(stringa) for stringa in group] for group in
                          corpus]

    # print("\nList_No_punct_numb:")
    # print(List_No_punct_numb)
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
