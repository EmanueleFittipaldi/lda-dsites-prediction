# lda-dsites-prediction



# Idea principale

L’idea di fondo è che, osservando ed estrapolando la distribuzione dei Topic da pagine nel dark-web, queste distribuzioni possano accumunare diversi siti, e, di conseguenza, possano in un qualche modo contribuire ad una predizione più accurata del crimine perpetrato in quel determinato sito, una volta fornito l’output di LDA ad una pipeline di predizione.


# Pipeline di lavoro

1. Usare un tool di NLP, per effettuare pre-processing del testo estratto da fonti quali: blog, forum, marketplace, in modo tale da avere un corpus uniforme.

2. Passare questo corpus ad LDA, per la quale configurazione impiegheremmo un algoritmo genetico per stabilire dei valori ottimali per gli hyperparameters (Alpha e Beta)  e k (numero di Topic da prendere in considerazione).

3. Si può in seguito, pluggare questi dati nella in una pipeline, al fine da verificare un effettivo aumento della accuratezza del rilevamento del crimine perpetrato su siti appartenenti al dark-web.
