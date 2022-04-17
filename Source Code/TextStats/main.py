'''
Se rimuovo le stopwords già adesso, questo non potrebbe andare ad influenzare il conteggio delle parole contenute in una frase?
E’ un problema quando arrivo alla fase di scartare tutte quelle linee che hanno meno di quattro parole. E.g. se ho un commento su più linee,
che per qualche motivo è stato diviso i più linee in malo modo, ovvero senza arrivare fino alla fine del rigo, se per esempio la frase era
di quattro parole, ma di cui una era una stopword, andando a togliere le stopword adesso, vado a perdemi il rigo.
Se invece non elimino adesso le stopwords, vado a salvare il rigo, però parallelamente sto permettendo anche a righi spuri di 4 parole
contenente una stopwords di sopravvivere. Bisogna osservare i righi spuri di lunghezza 4 parole, che contengono una stopwords, in media, quanti ne sono
cosi si può capire se, mantenendo le stopwords prima della fase dell’eliminazione dele righe spurie,
si introduce troppo errore oppure no.
Bisogna comparare anche quante righe andiamo a perdere delle righe dei commenti, rimuovendo le stopwords prima,
oppure dopo. Bisogna osservare anche se le righe spurie è meno probabile che contengano le stopwords

Quindi ricapitolando, voglio sapere:
- Quante righe spurie (righe di lunghezza 4) contengono stopwords, sul totale delle righe
- Calcolare, su tutte le righe di lunghezza <4, quante contengono una stopword'''



# Berlusconi market
# 1.5384615384615385% di righe contenenti meno di 4 parole, hanno una stop-word ed andranno perse (non è detto che siano tutte utili neanche)
# Righe totali:
# 74
#
# Numero di righe contenenti 4 o meno parole:
# 65
#
# Righe spurie di lunghezza 4 con almeno una stopwords:
# 1
#
# Righe legate probabilmente ai commenti:
# 9
# DarkMarket market
# 5.797101449275362% di righe contenenti meno di 4 parole, hanno una stop-word ed andranno perse (non è detto che siano tutte utili neanche)
# Righe totali:
# 96
#
# Numero di righe contenenti 4 o meno parole:
# 69
#
# Righe spurie di lunghezza 4 con almeno una stopwords:
# 4
#
# Righe legate probabilmente ai commenti:
# 27
# 1.7341040462427744% di righe contenenti meno di 4 parole, hanno una stop-word ed andranno perse (non è detto che siano tutte utili neanche)
# Righe totali:
# 429
#
# Numero di righe contenenti 4 o meno parole:
# 346
#
# Righe spurie di lunghezza 4 con almeno una stopwords:
# 6
#
# Righe legate probabilmente ai commenti:
# 83
#
# In media si perdono 3,0232223% righe su tutte le righe contenenti 4 o meno parole.

import gensim
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

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

def PieReviews():
    # Pie-chart righe totale e righe legate a commenti
    labels = 'Reviews', 'Total rows'
    sizes = [((righeTotali - righeMinoriUgualiDiQuattro) * 100) / righeTotali,
             100 - ((righeTotali - righeMinoriUgualiDiQuattro) * 100) / righeTotali]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # figure = plt.gcf()
    # figure.set_size_inches(8, 6)
    # plt.savefig('PieReviews.png')
    plt.show()


def PieTrashRow():
    # Pie-chart righe spurie su totale righe
    labels = "len(row) == 4 words with at least a stoppoword", "len(row) <= 4 words"
    sizes = [(righeSpurie * 100) / righeMinoriUgualiDiQuattro, 100 - (righeSpurie * 100) / righeMinoriUgualiDiQuattro]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # figure = plt.gcf()
    # figure.set_size_inches(8, 6)
    # plt.savefig('PieTrashRow.png')
    plt.show()

#DARKMARKET
#E:\ANITA-Dumps-Uncompressed\Sample 1\darkmarketsomqvzqfjudpd6t5eabgvvpplrbtzq6prervyogenlrlqd.onion\vendor\0c2c8cf0-423a-11ea-ba8c-61886ab94bd5\feedback.html

#BERLUSCONI_MARKET
#E:\ANITA-Dumps-Uncompressed\Sample 2\2019_08_20\2019_08_20_berlusconi_c=listings_a=vendor_v_id=2d9a6127f72fe3007ea30a1be8cbb104_tab=4.htm

#AGARTHA
#E:\ANITA-Dumps-Uncompressed\Sample 1\agartha_2020-03-31_18-36-01\agarthafw2cock27.onion\vendor\AmazingShop\ratings.html

file=open(r"E:\ANITA-Dumps-Uncompressed\Sample 1\agartha_2020-04-05_12-50-44\agarthafw2cock27.onion\vendor\coco86\ratings.html","r",encoding='utf8')
soup = BeautifulSoup(file, features="html.parser")
file.close()

#estraggo il testo dall'HTML
txToList= extractText(soup)
print("Testo estratto")
print(txToList)
print("\n")

#divido il testo in una stringa per riga
splitList=[]
splitList=txToList.split("\n")
print("\nText splitted into a list of strings:")
print(splitList)

#porto ogni stringa in minuscolo
textLowered=[]
for document in splitList:
        textLowered.append(document.lower())

#tokenizzo e rimuovo le parole minori di 4 caratteri e maggiori di 15 caratteri
new_corpus=[]
for document in textLowered:
    new_corpus.append(gensim.utils.simple_preprocess(document,deacc=True,min_len=4,max_len=15))

print("\nTokenized documents:")
print(new_corpus)

#rimuovo le liste vuote che non servono a niente ed interferiscono con i calcoli
abba = [x for x in new_corpus if x != []]
print("\nTolgo tutte le liste vuote, che intralciano solo i calcoli:")
print(abba)

righeTotali=0
for riga in abba:
    righeTotali=righeTotali+1
print("\nRighe totali:")
print(righeTotali)

import gensim
all_stopwords = gensim.parsing.preprocessing.STOPWORDS
righeSpurie=0
listaRigheSpurie=[]

for rigaSpuria in abba:
    x = check =  any(item in all_stopwords for item in rigaSpuria)
    if(len(rigaSpuria)==4 and x):
        righeSpurie=righeSpurie+1
        listaRigheSpurie.append(rigaSpuria)

righeMinoriUgualiDiQuattro=0

for rigaSpuria in abba:
    if(len(rigaSpuria)<=4):
        righeMinoriUgualiDiQuattro=righeMinoriUgualiDiQuattro+1
print("\nNumero di righe contenenti 4 o meno parole:")
print(righeMinoriUgualiDiQuattro)

print("\nRighe spurie di lunghezza 4 con almeno una stopwords:")
print(righeSpurie)

print("\nRighe legate probabilmente ai commenti:")
print(righeTotali-righeMinoriUgualiDiQuattro)

print("\n(queste sono le righe che mi sarei perso andando ad eliminare subito le stopwords, tenendo conto che applicherò un filtro dove elimino"
      "le righe che contengono meno di 4 parole)")
print(listaRigheSpurie)

print("\n")
print("rimuovendo le stopwords")


ListaSenzaStop=[]
from gensim.parsing.preprocessing import remove_stopwords
for document in listaRigheSpurie:
    for word in document:
        ListaSenzaStop.append(remove_stopwords(word))

print(ListaSenzaStop)


print("\nPercentuale delle righe lunghe 4 parole che contengono una stopword, su tutte quelle righe di lunghezza <=4")
print((righeSpurie/righeMinoriUgualiDiQuattro)*100)


PieReviews()
PieTrashRow()




