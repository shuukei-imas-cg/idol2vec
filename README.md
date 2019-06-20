# idol2vec
オンラインのサブカル百科辞典記事から取得した文章で分散表現を作りキャラクター性を定量的に扱おうという試み

「アイドルマスターシンデレラガールズ」(C)窪岡俊之 (C)BANDAI NAMCO Entertainment Inc. に登場する190人のアイドルに関するピクシブ百科事典の記事をスクレイピングし、各々の記事についてSWEM(Simple Word-Embedding-based Methods)によって文章の分散表現を作ります。この分散表現(ベクトル)により、アイドルのキャラクター性を定量的に扱えるようにしようというものです。


## Requirement
動作確認はUbuntu 18.04, Python 3.7.2、MeCab 0.996で行っています。

- Python (3.7以降)
- MeCab 
- mecab-ipadic-NEologd(強く推奨)


## Usage
### 環境構築
~~~
# pyenv, virtualenv等による独立したPython環境構築を強く推奨
pip -r requirements.txt
git clone
cd 
~~~


### 実行
~~~
# Pixiv百科事典から、names.csvに列挙した記事をスクレイピングする。
# data/ ディレクトリに(記事名).txtのテキストが生成される
python scrape.py names.csv

# im@sparqlよりユニット名・ユニット別名の一覧を取得しunits.jsonとして保存する
curl -o units.json "https://sparql.crssnky.xyz/spql/imas/query?force-accept=text%2Fplain&query=%0APREFIX%20schema%3A%20%3Chttp%3A%2F%2Fschema.org%2F%3E%0APREFIX%20rdf%3A%20%3Chttp%3A%2F%2Fwww.w3.org%2F1999%2F02%2F22-rdf-syntax-ns%23%3E%0APREFIX%20imas%3A%20%3Chttps%3A%2F%2Fsparql.crssnky.xyz%2Fimasrdf%2FURIs%2Fimas-schema.ttl%23%3E%0A%0ASELECT%20%20%3F%E3%83%A6%E3%83%8B%E3%83%83%E3%83%88%E5%90%8D%20%3F%E3%83%A6%E3%83%8B%E3%83%83%E3%83%88%E5%88%A5%E5%90%8D%20%0AWHERE%20%7B%0A%20%20%3Fs%20rdf%3Atype%20imas%3AUnit%3B%0A%20%20%20%20%20schema%3Aname%20%3F%E3%83%A6%E3%83%8B%E3%83%83%E3%83%88%E5%90%8D%3B%0A%20%20%20%20%20schema%3AalternateName%20%3F%E3%83%A6%E3%83%8B%E3%83%83%E3%83%88%E5%88%A5%E5%90%8D%3B%0A%7Dorder%20by(%3F%E3%83%A6%E3%83%8B%E3%83%83%E3%83%88%E5%90%8D)"

# mecab-ipadic-NEologdのパスの修正
(swem_from_article.py中のmecab_system_dicの値を、mecab-ipadic-NEologdのインストールパスにあわせて修正すること)

# 各記事の分散表現を作りpklファイルとして保存する
# python swem_from_article.py (アイドル名一覧のCSV) (Word2Vecモデルファイル) (保存先pklファイル名)
python swem_from_article.py names.csv corpus/ss-w2v.bin swem1.pkl

# python swem_predict (上で保存したpklファイル名) (アイドル指定:names.csvのインデックスで指定)
python swem_similar.py swem1.pkl 0
島村卯月
島村卯月(0): 0.0
本田未央(130): 0.07973699271678925
緒方智絵里(11): 0.08038537204265594
中野有香(1): 0.09118976444005966
姫川友紀(143): 0.09289788454771042
神谷奈緒(72): 0.09559869021177292
小日向美穂(10): 0.10004198551177979
多田李衣菜(76): 0.10141804814338684
相葉夕美(158): 0.101479172706604
諸星きらり(166): 0.10525113344192505
~~~

### ベクトルの加減算
pythonインタプリタ上で実行する例
~~~
import ngtpy
import pickle
with open("swem1.pkl", "rb") as f:
    idolvecs = pickle.load(f)

dim = 100
ngtpy.create(b"tmp", dim)
index = ngtpy.Index(b"tmp")
index.batch_insert(idolvecs[1])
index.save()

def similar_i(i):
	result = index.search(idolvecs[1][i], 10)
	for j, score in result:
		print(f"{idolvecs[0][j]}({j}): {score}")

def similar_v(v):
	result = index.search(v, 10)
	for j, score in result:
		print(f"{idolvecs[0][j]}({j}): {score}")

# ここまでコピペ

# 佐久間まゆ - 島村卯月 + 渋谷凛
diff = idolvecs[1][54] - idolvecs[1][0] + idolvecs[1][65]
similar_v(diff)
三船美優(79): 0.08821611106395721
矢口美羽(138): 0.1072150319814682
白菊ほたる(58): 0.11562131345272064
...
~~~


### TensorBoardによる可視化
~~~
#TensorBoardによる可視化を行う場合、追加で以下の依存ライブラリをインストールすること
pip install torch tensorboardX tensorflow

python vec2tfb.py swem1.pkl
# runsディレクトリに生成される

tensorboard --logdir=runs
# http://localhost:6006/ にブラウザでアクセスする
~~~


## 参考
- [SWEM: 単語埋め込みのみを使うシンプルな文章埋め込み](https://yag-ays.github.io/project/swem/)
- [学習済み分散表現をTENSORBOARDで可視化する (GENSIM/PYTORCH/TENSORBOARDX)](https://yag-ays.github.io/project/embedding-visualization/)


## Author
[@shuukei_imas_cg](https://twitter.com/shuukei_imas_cg)
