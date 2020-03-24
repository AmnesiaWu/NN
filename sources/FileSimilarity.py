# -*- coding:utf-8 -*-
import jieba
from gensim import models, corpora, similarities


def similarity():
    """
    :return: 测试文本与其他文本的相似度
    """
    puntuation = ["，", "。", "！", "‘", "”", "？", "·", "：", "“", "’", "—", "《", ";", "；"]
    doc0 = open(r'D:\py_pro\test\data\data0.txt', encoding='utf-8').read()
    doc1 = open(r'D:\py_pro\test\data\data1.txt', encoding='utf-8').read()
    doc2 = open(r'D:\py_pro\test\data\data2.txt', encoding='utf-8').read()
    test_doc = open(r'D:\py_pro\test\data\test_data.txt', encoding='utf-8').read()
    test_data = [word for word in jieba.cut(test_doc) if word not in puntuation]
    all_doc = list()
    all_doc.append(doc0)
    all_doc.append(doc1)
    all_doc.append(doc2)
    cut_doc = []
    for doc in all_doc:
        doc_list = [word for word in jieba.cut(doc) if word not in puntuation]
        cut_doc.append(doc_list)
    dictionary = corpora.Dictionary(cut_doc) # 生成字典
    corpus = [dictionary.doc2bow(doc) for doc in cut_doc] # 生成语料库
    test_corpus = dictionary.doc2bow(test_data) # 生成测试文档的语料库
    tfidf = models.TfidfModel(corpus)
    index = similarities.SparseMatrixSimilarity(corpus, num_features=len(dictionary.keys()))
    sim = index[tfidf[test_corpus]]
    res = sorted(enumerate(sim), key= lambda x:-x[1])
    print(res)
if __name__ == '__main__':
    similarity()