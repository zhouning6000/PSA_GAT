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
from tqdm import tqdm
from supar import Parser
dataset='Laptops'
file_name="D:\\papercode\\BiSyn_GAT_plus-main\\data\\V2\\Laptops\\all_con_new.json"
with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)

for d in tqdm(data):
    token = d['token']
    token = [tok.replace(u'\xa0', u'') for tok in token]
    d['token'] = token
# build vocab #{单词：频率}
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
file_name="D:\\papercode\\BiSyn_GAT_plus-main\\data\\V2\\Laptops\\all_con_new.json"
with open(file_name, 'r', encoding='utf-8') as f:
    data = json.load(f)
#all方面词
aspects=[]
for i in range(len(data)):
    for j in range(len(data[i]['aspects'])):
        #print(data[51]['aspects'][j])
        aspects.append(data[i]['aspects'][j]['term'])
#print(len(aspects))
#all句子
sentences=[]
for i in range(len(data)):
        for j in range(len(data[i]['aspects'])):
            sentences.append(data[i]['token'])
#print(len(sentences))
#all句子+方面词
sentences_aspects=[]
for i in range(len(sentences)):
    sentence_aspect=sentences[i]+[' ']+aspects[i]
    sentences_aspects.append(sentence_aspect)
sentences_aspects
#print(len(sentences_aspects))
#all序号
train_ids = []
for train_name in sentences_aspects[:2282]:
    train_id = sentences_aspects.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)
train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()
#print(len(train_ids))
test_ids = []
for test_name in sentences_aspects[2282:]:
    test_id = sentences_aspects.index(test_name)
    test_ids.append(test_id)
#print(test_ids)
random.shuffle(test_ids)
test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
ids
shuffle_sentences_aspects_list = []
for id in ids:
    shuffle_sentences_aspects_list.append(''.join(sentences_aspects[int(id)]))
shuffle_sentences_aspects_str = '\n'.join(shuffle_sentences_aspects_list)

f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_sentences_aspects_str)
f.close()



shuffle_sentences_aspects=[]
for id in ids:
    shuffle_sentences_aspects.append(sentences_aspects[int(id)])
    f = open('data/' + dataset + '_shuffle.txt', 'w')
    f.write(test_ids_str)
    f.close()
#print(shuffle_sentences_aspects)
label_set = set()
for d in data:
    for i in range(len(d['aspects'])):
            aspects = d['aspects'][i]['polarity']
            label_set.add(aspects)
#print(label_set)
label_list = list(label_set)
y=[]
for i in range(len(data)):
    for j in range(len(data[i]['aspects'])):
        #print(data[51]['aspects'][j])
        label=data[i]['aspects'][j]['polarity']
        #y.(data[i]['aspects'][j]['polarity'])
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
shuffle_y=[]
for id in ids:
    shuffle_y.append(y[int(id)])

shuffle_sentences_aspects_save = []
for i in range(len(shuffle_sentences_aspects)):
    temp = ""
    for j in range(len(shuffle_sentences_aspects[i])):
         temp = temp + " " + shuffle_sentences_aspects[i][j]
    shuffle_sentences_aspects_save.append(temp)
shuffle_y_save = []
for i in range(len(shuffle_y)):
    temp = ""
    for j in range(len(shuffle_y[i])):
        temp = temp + " " + str(shuffle_y[i][j])
    shuffle_y_save.append(temp)

shuffle_sentences_aspects_save_doc = '\n'.join(shuffle_sentences_aspects_save)
shuffle_y_save_doc = '\n'.join(shuffle_y_save)

ff = open('data/Laptops_shuffle.txt', 'w')
ff.write(shuffle_sentences_aspects_save_doc)
ff.close()
ff = open('data/Laptops_shuffle_y.txt', 'w')
ff.write(shuffle_y_save_doc)
ff.close()
# 1 计算build vocab #{单词：频率}
word_freq = {}
word_set = set()
for doc_words in shuffle_sentences_aspects:
    words = doc_words
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
for i in range(len(shuffle_sentences_aspects)):
    doc_words = shuffle_sentences_aspects[i]
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
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
vocab = list(word_set)
vocab_size = len(vocab)
vocab_str = '\n'.join(vocab)

f = open('data/Laptops_vocab.txt', 'w')
f.write(vocab_str)
f.close()
# 3{单词：对应的token的序列号}#{单词：【文档1，文档2】}
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)
# % 4 每个单词的序号
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
print("end")
definitions = []
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
#%%
f = open('data/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()
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
tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
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
_, embd, word_vector_map = loadWord2Vec(word_vectors)
string = '\n'.join(word_vectors)
f = open('data/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()
word_embeddings_dim = len(embd[0])
row_x = []
col_x = []
data_x = []
train_size=len(train_ids)
print(train_size)
#%%
for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_sentences_aspects[i]
    words = doc_words
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len
#%%
# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    train_size, word_embeddings_dim))

y = []
for i in range(train_size):
    doc_meta = shuffle_y[i]

    y.append(doc_meta)
y = np.array(y)
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_sentences_aspects[i + train_size]
    words = doc_words
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_y[i+train_size]
    ty.append(doc_meta)
ty = np.array(ty)
#print(ty)
word_vectors = np.random.uniform(-0.01, 0.01,(vocab_size, word_embeddings_dim))
for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector
row_allx = []
col_allx = []
data_allx = []
for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_sentences_aspects[i]
    words = doc_words
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))
ally=[]
for i in range(train_size):
    doc_meta = shuffle_y[i]
    ally.append(doc_meta)
for i in range(vocab_size):
    one_hot = [0 for l in range(3)]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)
window_size = 5
windows = []

for id in range(len(shuffle_sentences_aspects)):
    words = shuffle_sentences_aspects[id]
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)


word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights


# doc word frequency
doc_word_freq = {}

for id in range(len(shuffle_sentences_aspects)):
    doc_words = shuffle_sentences_aspects[id]
    words = doc_words
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

for i in range(len(shuffle_sentences_aspects)):
    doc_words = shuffle_sentences_aspects[i]
    words = doc_words
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_sentences_aspects) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)