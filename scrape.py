import time
import sys
import csv
import re
import pathlib
import requests
import urllib.parse
from bs4 import BeautifulSoup
base_url = "https://dic.pixiv.net/a/"


def normalize(text):
    text = re.sub(r"<[^>]*?>", "", text)  # htmlタグの除去
    text = re.sub(r"\n{1,5}", "\n", text)  # 不要な改行(1～5回)の除去
    text = text.replace("\u3000", "").replace("\xa0", "")  # 全角空白とnbspの除去
    return text


def get_and_save_article(name):
    pathlib.Path("./data").mkdir(exist_ok=True)
    fw = open("data/" + name + ".txt", "w")
    quoted_name = urllib.parse.quote(name)
    url = base_url + quoted_name
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html5lib")

    article = ""
    title = soup.find(id="article-name").string
    article += title + "\n"
    summary = soup.select("#content_title > div.summary")[0].string.strip()
    article += summary + "\n"

    body = soup.find("article").find(id="main").find(id="article-body")
    normalized_body = normalize(str(body))
    article += normalized_body
    fw.write(article)
    fw.close()


def main(name_list):
    f = open(name_list)
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        name = row[0]
        print(name)
        get_and_save_article(name)
        time.sleep(1)
    f.close()


if __name__ == '__main__':
    main(sys.argv[1])
