# -*- coding:utf-8 -*-
import sys
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = gensim.models.word2vec.LineSentence(sys.argv[1])
model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4, sg=1, hs=1, negative=0, iter=20)
model.wv.save_word2vec_format(sys.argv[2], binary=True)
