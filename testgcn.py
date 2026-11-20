import os
import json
from collections import Counter
import itertools
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
def loadWord2Vec(word_vectors):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    for line in word_vectors:
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i].strip())
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    return vocab, embd, word_vector_map
word_embeddings_dim = 1000

def preprocess_file2(file_name, dep_parser=None, con_parser=None, special_token='[N]'):
    print('Processing:', file_name)
    from tqdm import tqdm
    from supar import Parser
    if dep_parser is None:
        dep_parser = Parser.load('biaffine-dep-en') #Dependence表示句法分析树
    if con_parser is None:
        con_parser = Parser.load('crf-con-en')

    sub_len = len(special_token) #定义特殊字符的长度：3

    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f) #加载数据

    for d in tqdm(data):
        token = d['token']
        token = [tok.replace(u'\xa0', u'') for tok in token]
        d['token'] = token #分割字符串
    # 1 计算build vocab #{单词：频率}
    word_freq = {}
    word_set = set()
    for doc_words in data:
        words = doc_words['token']
        for word in words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
    print("end")
    vocab = list(word_set)
    vocab_size = len(vocab)
    # 2{单词：对应的token的序列号}#{单词：【文档1，文档2】}
    word_doc_list={}
    for i in range(len(data)):
        doc_words = data[i]['token']
        words = doc_words
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(i)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [i]
            appeared.add(word)
    print("end")
    # 3{单词：对应的token的序列号}#{单词：【文档1，文档2】}
    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)
    print(word_doc_freq)
    # % 4 每个单词的序号
    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i
    print("end")
    definitions = []
    #计算tf/idf的值

    for word in vocab:
        word = word.strip()
        synsets = wn.synsets(word)
        word_defs = []
        for synset in synsets:
            syn_def = synset.definition()
            word_defs.append(syn_def)
        word_des = ' '.join(word_defs)
        if word_des == '':
            word_des = '<PAD>'
        definitions.append(word_des)
    #计算tf/idf的值
    string = '\n'.join(definitions)
    tfidf_vec = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vec.fit_transform(definitions)
    tfidf_matrix_array = tfidf_matrix.toarray()
    print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))
    word_vectors = []
    for i in range(len(vocab)):
        word = vocab[i]
        vector = tfidf_matrix_array[i]
        str_vector = []
        for j in range(len(vector)):
            str_vector.append(str(vector[j]))
        temp = ' '.join(str_vector)
        word_vector = word + ' ' + temp
        word_vectors.append(word_vector)
    _, embd, word_vector_map = loadWord2Vec(word_vectors) #loadword2Vec将单词做成向量值

    #计算标签
    label_set = set()
    for d in data:
        for i in range(len(d['aspects'])):
            aspects = d['aspects'][i]['polarity']
            label_set.add(aspects)
    label_set = set()
    for d in data:
        for i in range(len(d['aspects'])):
            aspects = d['aspects'][i]['polarity']
            label_set.add(aspects)
    print(label_set)
    label_list = list(label_set)

    test_ids = []
    for d in data:
        test_id = data.index(d)
        test_ids.append(test_id)
    train_ids = [i for i in range(1454)]
    test_ids = [i + 1454 for i in range(409)]

    test_size = 409
    train_size = 1454
    train_size = len(train_ids)

    row_x = []
    col_x = []
    data_x = []
    for i in train_ids:
        doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
        doc_words = data[i]['token']
        words = doc_words
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_x.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    # x = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
    x = sp.csr_matrix((data_x, (row_x, col_x)),
                      shape=(train_size, word_embeddings_dim))
    print(label_set)
    #计算标签
    y = []
    for i in range(len(data)):
        for j in range(len(data[i]['aspects'])):
            # print(data[51]['aspects'][j])
            label = data[i]['aspects'][j]['polarity']
            y.append(data[i]['aspects'][j]['polarity'])
            one_hot = [0 for l in range(len(label_list))]
            label_index = label_list.index(label)
            one_hot[label_index] = 1
            y.append(one_hot)

if __name__ == '__main__':
    data_dir = 'data/V2'
    for data_set in ['Laptops']:#['Laptops', 'MAMS', 'Restaurants', 'Tweets']
        #for file_type in ['train', 'valid', 'test']:
        file_name = data_dir + '/' + data_set + '/' + 'all' + '_con_new.json'
        preprocess_file2(file_name)