import gensim
import os
import csv
import shutil
from bs4 import BeautifulSoup
from pprint import pprint
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

def getRatingsPaths(DatasetFolder):
    # Prendo tutti i percorsi inerenti alle cartelle chiamate vendor
    Vendors_folders = []
    for root, subdirs, files in os.walk(DatasetFolder):
        for d in subdirs:
            if d == "vendor":
                Vendors_folders.append(os.path.join(root, d))
    print("\nPercorsi delle cartelle contenenti i venditori\n")
    pprint(Vendors_folders)

    # Prendo tutti i percorsi di tutti i vendor contenuti nelle cartelle vendor
    # Ogni cartella è un vendor, il cui nome della cartella corrisponde al nome del vendor
    Vendors_names = []
    for VendorPath in Vendors_folders:
        p = os.listdir(VendorPath)  # Mi lista tutto quello che c'è in nella cartella vendor
        for file in p:
            if os.path.isdir(os.path.join(VendorPath, file)):
                Vendors_names.append(os.path.join(VendorPath, file))

    print("\nPercorsi delle cartelle di ogni venditore\n")
    pprint(Vendors_names)

    # Creo tutti i percorsi finali che mi portano ai ratings.html, dove sono contenute le recensioni
    Final_Paths = []
    for Vendor in Vendors_names:
        if(((os.listdir(Vendor)).pop(0)).endswith(".html")):
            Final_Paths.append(os.path.join(Vendor, (os.listdir(Vendor)).pop(0))) #in alcune cartelle non c'è solo ratings, ma c'è prima una icona, poi ratings
        else:                                                                     #quindi se il primo elemento non è un html, prendo il secondo. (Migliorabile)
            Final_Paths.append(os.path.join(Vendor, (os.listdir(Vendor)).pop(1)))

    print("\nPercorsi dei file delle recensioni per ogni venditore\n")
    pprint(Final_Paths)
    return Final_Paths

#------------------------------------------------------Estrazione delle righe---------------------------------------------------------------------------------
#Prendo tutti i percorsi dei file di ratings
RatingsPaths=getRatingsPaths("E:\ANITA-Dumps-Uncompressed")


#Converto ogni file di rating in un oggetto BeautifulSoup e lo salvo in una lista
SoupObjectsList=[]
for rating in RatingsPaths:
    f=open(rating, 'r',encoding='utf8')
    SoupObjectsList.append(BeautifulSoup(f, features="html.parser"))
    f.close()

if(len(SoupObjectsList)==len(RatingsPaths)):
    print("Sono stati rilevati "+ str(len(SoupObjectsList))+" venditori")
else:
    print("qualcosa è andato storto nella conversione dei file di rating in BeautifulSoup objects")



#Creo se non esiste un file CSV dove salverò le mie statistiche
stats= open('ExtractedStats.csv', mode='w')
stats_writer = csv.writer(stats, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
stats_writer.writerow(['Path', 'Total', 'R1','R2','Comments','R3',]) #Nomi delle colonne
CSV_row="" #inizializzo la riga che devo andare a scrivere

#Adesso, per ogni oggetto BeautifulSoup contenuto nella lista SoupObjectsList
#devo fare l'estrazione delle statistiche, così come ho fatto per un singolo file di rating
for soup in SoupObjectsList:

    # txToList = extractText(SoupObjectsList.pop(0))
    txToList = extractText(soup)

    # divido il testo in una stringa per riga
    splitList = []
    splitList = txToList.split("\n")
    print("\nText splitted into a list of strings:")
    print(splitList)

    # porto ogni stringa in minuscolo
    textLowered = []
    for document in splitList:
        textLowered.append(document.lower())

    # tokenizzo e rimuovo le parole minori di 4 caratteri e maggiori di 15 caratteri
    new_corpus = []
    for document in textLowered:
        new_corpus.append(gensim.utils.simple_preprocess(document, deacc=True, min_len=4, max_len=15))
    print("\nTokenized documents:")
    print(new_corpus)

    # rimuovo le liste vuote che non servono a niente ed interferiscono con i calcoli
    abba = [x for x in new_corpus if x != []]
    print("\nTolgo tutte le liste vuote, che intralciano solo i calcoli:")
    print(abba)
    # -------------------------------------------------------------Generazione delle statistiche------------------------------------------------------------
    '''RIGHE TOTALI:'''
    righeTotali = 0
    for riga in abba:
        righeTotali = righeTotali + 1
    print("\nRighe totali:")
    print(righeTotali)
    CSV_row = str(RatingsPaths.pop(0))+","+str(righeTotali) + "," #inizio a comporre la riga che andrò a scrivere nel CSV


    all_stopwords = gensim.parsing.preprocessing.STOPWORDS
    righeSpurie = 0
    listaRigheSpurie = []
    for rigaSpuria in abba:
        x = check = any(item in all_stopwords for item in rigaSpuria)
        if (len(rigaSpuria) == 4 and x):
            righeSpurie = righeSpurie + 1
            listaRigheSpurie.append(rigaSpuria)
    righeMinoriUgualiDiQuattro = 0
    for rigaSpuria in abba:
        if (len(rigaSpuria) <= 4):
            righeMinoriUgualiDiQuattro = righeMinoriUgualiDiQuattro + 1
    print("\nNumero di righe contenenti 4 o meno parole:")
    print(righeMinoriUgualiDiQuattro)
    CSV_row = CSV_row + str(righeMinoriUgualiDiQuattro) + ","


    print("\nRighe spurie di lunghezza 4 con almeno una stopwords:")
    print(righeSpurie)
    CSV_row = CSV_row + str(righeSpurie) + ","


    print("\nRighe legate probabilmente ai commenti:")
    print(righeTotali - righeMinoriUgualiDiQuattro)
    CSV_row = CSV_row + str(righeTotali - righeMinoriUgualiDiQuattro) + ","


    print(
        "\n(queste sono le righe che mi sarei perso andando ad eliminare subito le stopwords, tenendo conto che applicherò un filtro dove elimino"
        "le righe che contengono meno di 4 parole)")
    print(listaRigheSpurie)


    print("\n")
    print("rimuovendo le stopwords")

    ListaSenzaStop = []
    from gensim.parsing.preprocessing import remove_stopwords

    for document in listaRigheSpurie:
        for word in document:
            ListaSenzaStop.append(remove_stopwords(word))

    print(ListaSenzaStop)

    print(
        "\nPercentuale delle righe lunghe 4 parole che contengono una stopword, su tutte quelle righe di lunghezza <=4")
    print((righeSpurie / righeMinoriUgualiDiQuattro) * 100)

    temp = ((righeTotali - righeMinoriUgualiDiQuattro) * 100) / righeTotali

    CSV_row = CSV_row + str(temp) + "%"
    stats_writer.writerow([CSV_row])
    # PieReviews()
    # PieTrashRow()





