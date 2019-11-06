

import requests
from bs4 import BeautifulSoup

from urllib import request
import urllib
import random
from urllib.request import Request, urlopen

def iteration (max_pages):
    pages = 0

import requests
from bs4 import BeautifulSoup

from urllib import request
import urllib
import random
from urllib.request import Request, urlopen

def iteration (max_pages):
    pages = 0
    while pages <= max_pages :
        url = 'https://pixabay.com/fr/images/search/pomme%20rouge/?pagi='+str(pages)# parcourir toute les page du site 
        opening = requests.get(url)
        save = opening.text


        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
       # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            #print(links)
            if 'https://pixabay.com/fr/images/search/pomme%20rouge/?pagi='+str(pages) in links:
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom,"\n")
                if nom.count('Pomme Rouge') == 1 and nom.count('Fruits') == 1: #filtrer les données a récuperer 
                    name = random.randrange(1, 180)

                    full_name = ("./pomme_rouge/" + str(name)+"Pomme Rouge" + '.jpg') # le dossier ou enregistrer l'image 



                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)






    pages = pages + 1
iteration(1)

def int_pom_rouge(max_pages):
    pages = 0
    while pages <= max_pages:
        url = ('https://pixabay.com/fr/images/search/ananas/?pagi='+str(pages))

        # if pages%2 == 0:

        # else:
        # url = 'https://pixabay.com/images/search/pomme/'+str(pages)

        opening = requests.get(url)

        save = opening.text
        # print(save)

        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
        # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            # print(links)
            if 'https://pixabay.com/fr/images/search/ananas/?pagi='+str(pages) in links:
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom, "\n")
                if nom.count('Ananas') == 1 and  nom.count('Fruits') == 1 :
                    name = random.randrange(1, 180)

                    full_name = ("./ananas/" + str(name) + "Ananas" + '.jpg')

                    # urllib.request.urlretrieve(src, full_name)

                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)



    pages = pages + 1


int_pom_rouge(2)

def int_banane(max_pages):
    pages = 1
    while pages <= max_pages:
        url = ('https://pixabay.com/fr/images/search/banane/?pagi='+str(pages))  # incrementer les page
        print(url)

        # if pages%2 == 0:

        # else:
        # url = 'https://pixabay.com/images/search/pomme/'+str(pages)

        opening = requests.get(url)

        save = opening.text
        # print(save)

        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
        # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            # print(links)
            if 'https://pixabay.com/fr/images/search/banane/?pagi='+str(pages) in links: # incrementer les page
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom, "\n")
                if nom.count('Banane') == 1 and nom.count('Jaune') == 1  : # filtrer les données
                    name = random.randrange(1, 180)

                    full_name = ("./banane/" + str(name) + "Banane" + '.jpg')

                    # urllib.request.urlretrieve(src, full_name)

                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)



    pages = pages + 1


int_banane(5)
    while pages <= max_pages :
        url = 'https://pixabay.com/fr/images/search/pomme%20rouge/?pagi='+str(pages)
        opening = requests.get(url)
        save = opening.text


        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
       # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            #print(links)
            if 'https://pixabay.com/fr/images/search/pomme%20rouge/?pagi='+str(pages) in links:
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom,"\n")
                if nom.count('Pomme Rouge') == 1 and nom.count('Fruits') == 1: #filtrage
                    name = random.randrange(1, 180)

                    full_name = ("./pomme_rouge/" + str(name)+"Pomme Rouge" + '.jpg')



                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)






    pages = pages + 1
iteration(1)

def int_pom_rouge(max_pages):
    pages = 0
    while pages <= max_pages:
        url = ('https://pixabay.com/fr/images/search/ananas/?pagi='+str(pages))

        # if pages%2 == 0:

        # else:
        # url = 'https://pixabay.com/images/search/pomme/'+str(pages)

        opening = requests.get(url)

        save = opening.text
        # print(save)

        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
        # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            # print(links)
            if 'https://pixabay.com/fr/images/search/ananas/?pagi='+str(pages) in links:
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom, "\n")
                if nom.count('Ananas') == 1 and  nom.count('Fruits') == 1 :
                    name = random.randrange(1, 180)

                    full_name = ("./ananas/" + str(name) + "Ananas" + '.jpg')

                    # urllib.request.urlretrieve(src, full_name)

                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)



    pages = pages + 1


int_pom_rouge(2)

def int_banane(max_pages):
    pages = 1
    while pages <= max_pages:
        url = ('https://pixabay.com/fr/images/search/banane/?pagi='+str(pages))  # incrementer les page
        print(url)

        # if pages%2 == 0:

        # else:
        # url = 'https://pixabay.com/images/search/pomme/'+str(pages)

        opening = requests.get(url)

        save = opening.text
        # print(save)

        soup = BeautifulSoup(save, 'html.parser')
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))
        # print(class_item)

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            # print(links)
            if 'https://pixabay.com/fr/images/search/banane/?pagi='+str(pages) in links: # incrementer les page
                continue

            opening = requests.get(links)
            save = opening.text
            # print(save)

            soup = BeautifulSoup(save, 'html.parser')
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})
            #  print(main_site)

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt')
                print(nom, "\n")
                if nom.count('Banane') == 1 and nom.count('Jaune') == 1  : # filtrer les données
                    name = random.randrange(1, 180)

                    full_name = ("./banane/" + str(name) + "Banane" + '.jpg')

                    # urllib.request.urlretrieve(src, full_name)

                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)



    pages = pages + 1


int_banane(5)
