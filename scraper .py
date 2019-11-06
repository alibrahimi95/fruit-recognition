import requests
from bs4 import BeautifulSoup
import urllib.request
import random
import os

url = 'https://pixabay.com/fr/images/search/'


def scrap(search, max_pages):

    dirName = "D:/M2Ise/img/"+str(search)
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    pages = 1
    while pages <= max_pages:
        urlS =url +str(search)+'/?pagi='+str(pages)  # incrementer les page

        opening = requests.get(urlS)

        save = opening.text

        soup = BeautifulSoup(save)
        class_item = (div.a for div in soup.find_all('div', {'class': 'item'}))

        for l in class_item:
            links = 'https://pixabay.com/' + str(l.get("href"))
            # print(links)
            if urlS in links: # incrementer les page
                continue

            opening = requests.get(links)
            save = opening.text

            soup = BeautifulSoup(save)
            main_site = soup.find_all('img', {'itemprop': 'contentURL'})

            for t in main_site:
                src = t.get('src')
                nom = t.get('alt').lower()
                tab = search.split()
                if nom.count(tab[0]) == 1 and nom.count(tab[1]) == 1  : # filtrer les données
                    name = random.randrange(1, 180)

                    full_name = (dirName+"/" + str(name) + '.jpg')

                    # urllib.request.urlretrieve(src, full_name)

                    class AppURLopener(urllib.request.FancyURLopener):
                        version = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.69 Safari/537.36"

                    urllib._urlopener = AppURLopener()

                    urllib._urlopener.retrieve(src, full_name)
            pages = pages + 1

scrap("pomme rouge",2)
scrap("banane jaune",2)
scrap("ananas fruit",2)
