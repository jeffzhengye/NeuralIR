r"""
Word2Vec Model
==============

Introduces Gensim's Word2Vec model and demonstrates its use on the Lee Corpus.

"""

import logging
import os
import sys

from gensim.models import word2vec
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from gensim.models.word2vec import PathLineSentences

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if len(sys.argv) < 2:
    print('Needs 1 arguments - the corpus file')
    exit(0)

corpus_file = sys.argv[1]

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                parts = line.split(' ', 1)
                yield  parts[1].split(' ')


sentences = MySentences(corpus_file)  # a memory-friendly iterator


# class MyCorpus(object):
#     def __iter__(self):
#         corpus_path = datapath(r'C:\Users\23237\Desktop\word2vec_data\word2vec_data\text8截取.txt')
#         for line in open(corpus_path, encoding='utf-8'):
#             print(line)
#             yield utils.simple_preprocess(line)
#
# sentences = MyCorpus()
print("开始训练模型")
model = gensim.models.Word2Vec(sentences, min_count=0, window=10, negative=10,sample=1e-4, size=300, workers=5, sg=0)
print("模型训练结束")

# 保存模型
model.save('trec_corpus.model')
print("模型已保存！")
