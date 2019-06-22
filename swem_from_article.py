import csv
import logging
import re
import sys
import numpy as np
import json
import neologdn
from swem import MeCabTokenizer
from swem import SWEM
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
mecab_system_dic = "/opt/mecab/dic/mecab-ipadic-neologd"


class MeCabTokenizerWithStopWord(MeCabTokenizer):
    """
    swem.MeCabTokenizerから、ストップワードを扱えるようオーバーライドしたもの
    """
    def __init__(self, mecab_args="", nouns=set()):
        super().__init__(mecab_args)
        # self.tagger = MeCab.Tagger(mecab_args)
        self.nouns = nouns
        self.tagger.parse("")

    def tokenize(self, text):
        wakati_list = self.tagger.parse(text).strip().split(" ")
        wakati_list = remove_proper_noun(self.nouns, wakati_list)
        return wakati_list


def remove_proper_noun(nouns: set, wakati_list: list) -> list:
    """
    わかち書きされた形態素のリストを入力として、nounsに含まれるトークンを除外したリストを返す
    """
    new_wakati_list = []

    for token in wakati_list:
        if token not in nouns:
            # nounsに含まれないトークンのみを返すようにする
            new_wakati_list.append(token)

    return new_wakati_list


def normalize(text: str) -> str:
    """
    テキストの正規化
    """
    text = text.replace("\n", " ").strip()  # 改行を除去して1行の長いテキストとみなす
    text = re.sub(r'(\d)([,.])(\d+)', r'\1\3', text)  # 数字の桁区切りの除去
    text = re.sub(r'\d+', '0', text)  # 数字をすべて0に統一
    text = neologdn.normalize(text)  # 全角・半角の統一と重ね表現の除去
    return text


def build_nouns_set(name_list_file):
    """
    名前・愛称・ユニット名から、除外する固有名詞のセット(集合)を用意する
    """
    with open(name_list_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        nouns = []
        for row in reader:
            name = row[0]
            if "(シンデレラガールズ)" in name:
                name = name.replace("(シンデレラガールズ)", "")
            nouns.append(name)
            nickname = row[1]
            nouns.append(nickname)
            if nickname in name:
                nouns.append(name.replace(nickname, ""))  # とりあえず、name - nickname の部分文字列を名字とみなす

    # ユニット名も追加する
    with open("units.json") as fj:
        units_json = json.load(fj)
        for unit in units_json["results"]["bindings"]:
            for k in unit:
                nouns.append(unit[k]["value"])

    nouns_set = set(nouns)
    return nouns_set


def main(name_list_file, w2v_model, pkl_filename):
    f = open(name_list_file)
    reader = csv.reader(f)
    header = next(reader)
    nouns_set = build_nouns_set(name_list_file)

    # Word2Vecモデルのロード時、モデルファイルの形式にあわせてロード手順を変えること

    # KeyedVectorsの場合(save_word2vec_formatで保存したもの)
    from gensim.models import KeyedVectors
    w2v = KeyedVectors.load_word2vec_format(w2v_model, binary=True)  # .bin形式の場合
    # w2v = KeyedVectors.load_word2vec_format(w2v_model, binary=False)  # .txt形式の場合

    # Word2Vec.saveで保存したもの
    # import gensim.models.doc2vec as doc2vec
    # w2v = doc2vec.Doc2Vec.load("pixiv/doc2vec.model")

    # fastTextモデルの場合
    # from gensim.models.wrappers.fasttext import FastText
    # w2v = FastText.load_fasttext_format('pixiv/fasttext-model.bin')

    tokenizer = MeCabTokenizerWithStopWord(mecab_args=f"-O wakati -d {mecab_system_dic}", nouns=nouns_set)
    swem = SWEM(w2v, tokenizer)

    names = []
    vecs = []
    attributes = {}
    for row in reader:
        name = row[0]
        nickname = row[1]
        attribute = row[2]
        if attribute not in attributes:
            attributes[attribute] = []

        with open("data/" + name + ".txt") as n:
            text = n.read()
            text = normalize(text)

            vec = swem.average_pooling(text)
            # vec = swem.max_pooling(text)

            if "(シンデレラガールズ)" in name:
                name = name.replace("(シンデレラガールズ)", "")
            names.append(name)
            vecs.append(vec)
            attributes[attribute].append(vec)

    # 各属性の平均ベクトルを作る
    for key in attributes:
        ave = np.average([v for v in attributes[key]], axis=0)
        names.append(key)
        vecs.append(ave)

    f.close()
    idolvecs = [names, vecs]

    # pklファイルへ保存
    with open(pkl_filename, 'wb') as pkl:
        pickle.dump(idolvecs, pkl)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
    # main("names.csv", "corpus/ss-w2v.bin", "swem1.pkl")
