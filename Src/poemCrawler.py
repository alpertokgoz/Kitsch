import requests
import bs4 as bs
import codecs

def getPoemLinks():
    retval=[]
    for i in range(0,10):
        pageUrl='http://www.antoloji.com/kucuk-iskender/siirleri/ara-/sirala-/sayfa-%s/' % i
        response = requests.get(pageUrl)
        print(pageUrl)
        soup = bs.BeautifulSoup(response.text, "lxml")
        links = soup.find_all('td', {'class': 'liste_border'})
        for link in links:
            for a_elm in link.find_all("a"):
                if 'kucuk-iskender/siirleri' in a_elm.attrs['href']:
                    continue
                retval.append("http://www.antoloji.com" + a_elm.attrs['href'])
    print('%s poem found'  % len(retval))
    return retval

def getPoemText(pLink):
    print(pLink)
    response = requests.get(pLink)
    print(response.text.encode('utf-8'))
    soup = bs.BeautifulSoup(response.text.replace('<br>', "\n").replace("</br>", ""))
    found = soup.find("font", class_="Siir_metin").get_text()

    return found


def readPoemsFromFile():
    import codecs
    with codecs.open("kucukiskender.txt", "r", "UTF-8") as f:
        return [e.replace("***") for e in f.readlines()]


def main():
    poemTexts = []
    for link in getPoemLinks():
        poemTexts.append(getPoemText(link))
    with codecs.open('kucukiskender.txt', "w", "UTF-8") as f:
        for p in poemTexts:
            f.write(p)
            f.write('\n***\n')
    return poemTexts

if __name__=='__main__':
    main()

