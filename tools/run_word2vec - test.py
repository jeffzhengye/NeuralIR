r"""
Word2Vec Model
==============

Introduces Gensim's Word2Vec model and demonstrates its use on the Lee Corpus.

"""

import logging
from gensim.models.word2vec import PathLineSentences, Word2Vec
from gensim.models import word2vec
from gensim.test.utils import datapath
from gensim import utils
import gensim.models


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# class MyCorpus(object):
#     def __iter__(self):
#         corpus_path = datapath(r'D:\纵向项目\neural-ranking-drmm-master\neural-ranking-drmm-master\data\trec_corpus.txt')
#         for line in open(corpus_path, encoding='utf-8'):
#             # print(line)
#             yield utils.simple_preprocess(line)
#
# sentences = MyCorpus()
# print(sentences)
input_dir = 'D:\纵向项目\preprocessing\word'
print("开始训练模型")
model = Word2Vec(PathLineSentences(input_dir), min_count=0, window=10, size=300, sg=0,sample=1e-4,negative=10)
# model = gensim.models.Word2Vec(sentences, min_count=0, window=10, size=300, sg=0,sample=1e-4,negative=10)
print("模型训练结束")

# 保存模型
model.save('trec_corpus_stemmed.model')
print("模型已保存！")

# 加载模型
# model = word2vec.Word2Vec.load('text8截取.model')
# print("模型已加载！")
# model.most_similar(positive=['woman', 'king'], negative=['man'], topn=3)
# print(model.most_similar("anarchism", topn=3))
# print(model.similarity("anarchism", "anarchists"))
# print("词类比")
# model.wv.accuracy(r'C:\Users\23237\Desktop\word2vec_data\word2vec_data\类比_questions-words.txt')
# print("词相似度")
# model.evaluate_word_pairs(datapath(r'C:\Users\23237\Desktop\word2vec_data\word2vec_data\词相似度_wordsim-353.txt'))