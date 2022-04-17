import pickle
from collections import defaultdict
from bs4 import BeautifulSoup

#IMPORTING THE FUNCTIONS NEEDED TO PRE-PROCESS THE TEXT
from functions import getRatingsPaths
from functions import extractText
from functions import splitAndLower
from functions import TokenizingCorpus
from functions import delUnder4Words
from functions import noPuncNoNumb
from functions import errataCorrige
from functions import deContract
from functions import singAndLemm
from functions import reStopWords

# Con questo .py cerco di trovare un modo con cui riuscire ad esptrapolare le review da qualsiasi sito del darkweb.
# Alla luce di ciò, non è più possibile andare ad ancorarsi alla struttura specifica di un sito in particolare, per identiricare
# un pattern ricorrente nella disposizione degli elementi HTML, come si fa di solito durante il processo di scraping.
# L'approccio utilizzato è il seguente. Mediante questa funzione extractText al quale si passa un oggetto di tipo BeautifulSoup
# riusciamo ad estrarre tutto il testo, esente da tag html, presente nel sito. Questo è il massimo che si riesce a fare, adottando
# un approccio generico. Per cercare di filtrare il testo, estrapolando solo i commenti, ho fatto affidamento al preprocessing.
# In particolare, quello che è stato fatto è, prima di tutto prendere il documento generato dall'estrazione del testo presente in
# tutta la pagina, dividerlo in una lista di stringhe, dove ogni stringa corrisponde ad una riga del documento iniziale.

# -------------------------------------------------------------------------------FUNCTIONS----------------------------------------------------------------------

i=0
LISTA_CORPUS_PROCESSATI=[]

#-----------------------------------------------------Step 0 - Opening the file and extracting the HTML as a BeautifulSoup Object--------------------------------

#Prendo tutti i percorsi dei file di ratings a partire dalla root del dataset
RatingsPaths=getRatingsPaths("E:\ANITA-Dumps-Uncompressed")

#SoupObjectsList conterrà ogni file delle recensioni convertito in un oggetto BeautifulSoup
SoupObjectsList=[]
for rating in RatingsPaths:
    f=open(rating, 'r',encoding='utf8')
    SoupObjectsList.append(BeautifulSoup(f, features="html.parser"))
    f.close()

#Mi assicuro soltanto che gli oggetti BeautifulSoup siano esattamente di numero uguale ai path inizalmente rilevati
if(len(SoupObjectsList)==len(RatingsPaths)):
    print("Sono stati rilevati "+ str(len(SoupObjectsList))+" venditori")
else:
    print("qualcosa è andato storto nella conversione dei file di rating in BeautifulSoup objects")

#Adesso, per ogni oggetto BeautifulSoup vado ad eseguire tutti i passi della pipeline. Il testo preprocessato di ogni file
#viene salvato nella lista LISTA_CORPUS_PREPROCESSATI
for soup in SoupObjectsList:

    # -----------------------------------------------------Step 1 - Getting the text from an HTML file---------------------------------------------------------------
    txToList = extractText(soup)
    # print("Testo estratto:")
    # print(txToList)
    # print("\n")

    # -----------------------------------------------------Step 2 - Splitting the document into a List of strings----------------------------------------------------
    # Here I am splitting the whole string extracted from the website into little chunks. I am splitting It into several lines, everytime a "\n" is encountered.
    # Then I lower each line. From this step, we obtain the raw string in a workable format, from which we can apply further processing.
    filteredCorpus = splitAndLower(txToList)

    # ----------------------------------------Step 3 - Tokenizing every string and removing tokens shorter than 4 characters and longer than 15 character------------
    # Here I remove all the words of length <4 or >15. This will empty some lists, hence I need to remove them.
    TextExtracted = TokenizingCorpus(filteredCorpus)

    # ----------------------------------------Step 4 - Creating a new List, keeping only documents with at least 4 words-------------------------------------------
    # Here I apply the observation that all the lines that are less than 4 words, have a high probability of being just trash lines, that I need to remove in order
    # to extract the text from the reviews. So, here, I am creating another list where I save only the lists which have a length >=4
    prettyList = delUnder4Words(TextExtracted)

    # ----------------------------------------Step 5  -  Eliminare la punteggiatura e numeri----------------------------------------------------------------
    List_No_punct_numb = noPuncNoNumb(prettyList)

    # ----------------------------------------Step 6  -  Spelling correction--------------------------------------------------------------------------------
    List_spellChecked = errataCorrige(List_No_punct_numb)
    # print("\n")
    # print("List_spellChecked")
    # print(List_spellChecked)
    # print("\n")

    # ----------------------------------------Step 7  -  Espansione delle contrazioni-----------------------------------------------------------------------
    List_deContracted = deContract(List_spellChecked)

    # ----------------------------------------Step 8  -  Singolarizzazione e lemmatizzazione----------------------------------------------------------------------------------
    List_singAndLem = singAndLemm(List_deContracted)

    # ----------------------------------------Step   -  StopWords removal-------------------------------------------------------------------------------------------------
    List_final = reStopWords(List_singAndLem)

#-----------------------NON SO SE SERVE--------------#
    # in questo dizionario sono presenti le parole con assegnata la frequenza
    frequency = defaultdict(int)
    for text in List_final:
        for token in text:
            frequency[token] += 1
    # print("\n")
    #print(frequency)

    # manteniamo soltanto le parole che appaiono più di una volta e meno di 30 volta
    processed_corpus = [[token for token in text if frequency[token] > 1 and frequency[token] < 30] for text in
                        List_final]
    #pprint.pprint(processed_corpus)


    # Aggiorno le frequenze in questo dizionario sono presenti le parole con assegnata la frequenza
    frequency = defaultdict(int)
    for text in processed_corpus:
        for token in text:
            frequency[token] += 1
    # print("\n")
    # print(frequency)
# -----------------------NON SO SE SERVE--------------#

    #aggiungo il corpus processato alla lista che poi andrò ad utilizzare alla fine
    #per generare il Bag of words
    print("documento processato numero "+str(i))
    i+=1
    print(processed_corpus)

    for sentence in processed_corpus:
        LISTA_CORPUS_PROCESSATI.append(sentence)

print("\nLISTA_CORPUS_PROCESSATI")
print(LISTA_CORPUS_PROCESSATI)


# PER SALVARE il lavoro di preprocessing sul disco per non doverlo ripetere
with open('Serialized_Processed_Corpora', 'wb') as fp:
    pickle.dump(LISTA_CORPUS_PROCESSATI, fp)