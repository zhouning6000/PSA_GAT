import random
import heapq
import os
import json
import torch
import math
import numpy as np 
from transformers import BertTokenizer
import copy 
import random 
import itertools 
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from spans import *

class ABSA_Dataset(Dataset):
    def __init__(self, args, file_name, vocab, tokenizer):
        super().__init__()

        # load raw data
        with open(file_name,'r',encoding='utf-8') as f:
            raw_data = json.load(f)

            if args.need_preprocess:
                raw_data = self.process_raw(raw_data)
                new_file_name = file_name.replace('.json','_con.json')
                with open(new_file_name, 'w', encoding='utf-8') as f:
                    json.dump(raw_data,f)
                print('Saving to:', new_file_name)

        self.data = self.process(raw_data, vocab, args, tokenizer)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    
    def process_raw(self, data):
        # get parserd data
        # we already provide here
        pass
    

    def process(self, data, vocab, args, tokenizer):
        token_vocab = vocab['token']
        pol_vocab = vocab['polarity']

        processed = []
        max_len = args.max_len  #定义最大的截距
        CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        sub_len = len(args.special_token)


        for d in data:
            tok = list(d['token'])
            if args.lower:
                tok = [t.lower() for t in tok]
            #bert 句子编码 #bert编号
            text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer)
    #截取指定长度的值
            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]
#长度
            length = word_mapback[-1] + 1
            # tok = tok[:length]
            bert_length = len(word_mapback)
#取深度的头 语义节点
            dep_head = list(d['dep_head'])[:length]

            # map2id 
            # tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # con_head 结构节点
            con_head = d['con_head']
            #con_mapnode
            con_mapnode = d['con_mapnode']
            # con_path_dict# 表示到root祖宗点的路径  #con_children表示 儿子点集合
            con_path_dict, con_children = get_path_and_children_dict(con_head)
            # con_mapnode
            mapback = [ idx for idx ,word in enumerate(con_mapnode) if word[-sub_len: ]!= args.special_token]
#layer代表第一层有几个节点，第二层有几个节点号第三层有几个节点号， influence——range 代  输入是代表
            layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback)
#
            spans = form_spans(layers, influence_range, length, con_mapnode)

            adj_i_oneshot = head_to_adj_oneshot(dep_head, length, d['aspects'])#将dep_head中的依存语法数的头和方面集标注上1

            cd_adj = np.ones((length,length))
            if args.con_dep_conditional:
                father = 1
                if father in con_children and [con_mapnode[node] for node in con_children[father]].count('S[N]') > 1 and con_mapnode[father] == 'S[N]':
                    cd_span = spans[node2layerid[father]+1]
                    cd_adj = get_conditional_adj(father, length, cd_span, con_children, con_mapnode)

            adj_i_oneshot = adj_i_oneshot * cd_adj #按元素相乘

            # aspect-specific
            bert_sequence_list = [] #方面集内容
            bert_segments_ids_list = []  #方面集标签
            label_list = [] #标签
            aspect_indi_list = [] #方面在list的位置
            aspect_indi_list_detai=[]
            select_spans_list = [] #选择跨度序列
            aspect_indi_all=[0]*length#标注本句中所有的方面词
            for aspect in d['aspects']: #选取一个方面集合的子集
                asp = list(aspect['term']) #选择子集中的【‘term’】
                asp_bert_ids, _, _ = text2bert_id(asp, tokenizer) #方面集的bert表示
                bert_sequence = CLS_id  + text_raw_bert_indices +  SEP_id + asp_bert_ids + SEP_id  #开始字符+全部句子的序号标识+分隔符+方面字+分隔符
                bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids) +1) #标识句子字符和方面字符
                bert_sequence = bert_sequence[:max_len+3]# 限制长度
                bert_segments_ids = bert_segments_ids[:max_len+3] #截断长度
                label = aspect['polarity'] #取方面集的属性
                aspect_indi = [0] * length #方面集的指引
                for pidx in range(aspect['from'], aspect['to']):
                    aspect_indi[pidx] = 1
                    aspect_indi_all[pidx]=1
                aspect_indi_list_detai.append(aspect_indi)
                label = pol_vocab.stoi.get(label) #取标签的内容
                aspect_range = list(range(mapback[aspect['from']], mapback[aspect['to']-1] + 1)) #方面集的范围
                con_lca = find_inner_LCA(con_path_dict, aspect_range) #取方面集的公共父集
                select_spans, span_indications = form_aspect_related_spans(con_lca, spans, con_mapnode, node2layerid, con_path_dict)# con_lca 的路径在span中的体现
                select_spans = select_func(select_spans, args.max_num_spans, length)
                select_spans = [[ x+ 1 for x in span] for span in select_spans]
                label_list.append(label)
                aspect_indi_list.append(aspect_indi)
                bert_sequence_list.append(bert_sequence)
                bert_segments_ids_list.append(bert_segments_ids)
                select_spans_list.append(select_spans)
            #*********************#
            p = []  # 权重值
            n = 0
            mask_all = {}#表示第几个方面的方面在句子中的序号
            level=4
            mask_all_list = []

            adj_matrix_distance_TF=torch.zeros( len(aspect_indi_list_detai), len(aspect_indi_list_detai[0]))
            adj_matrix_distance_idf=torch.zeros( len(aspect_indi_list_detai), len(aspect_indi_list_detai[0]))

            for i in range(len(aspect_indi_list_detai)):
                # search 一下mask的点
                mask = []
                for j in range(len(aspect_indi_list_detai[i])):
                    if (aspect_indi_list_detai[i][j] == 1):
                        mask.append(j)
                mask_all[i] = (mask)
            for i, j in mask_all.items():
                mask_all_list.append(j)
            map_AS = torch.LongTensor([m for m_list in mask_all_list for m in m_list])
            for i in range(len(aspect_indi_list_detai)):
                for j in range(len(aspect_indi_list_detai[i])):
                    down = mask_all[i][0]
                    up = mask_all[i][-1]
                    if j < down:
                        adj_matrix_distance_TF[i, j] =1 - (down - j) / len(aspect_indi_list_detai[i])
                    if j >= down and j <= up:
                        adj_matrix_distance_TF[i, j] = 1
                    if j >= down:
                        adj_matrix_distance_TF[i, j] = 1 - (j - down) / len(aspect_indi_list_detai[i])
            for i in range(len(aspect_indi_list_detai)):
                for j in range(len(aspect_indi_list_detai[i])):  # j代表这一个节点
                    if (j not in map_AS):
                        distance = {}
                        num_d = 0  # 下面是这个节点对应的每个aspect的距离
                        for id, item in mask_all.items():
                            distance[id] = min(abs(item[0] - j), abs(item[-1] - j))
                            if distance[id] - level <= 0:  # 距离比较小的时候
                                num_d = num_d + 1
                        adj_matrix_distance_idf[i, j] = math.log(len(aspect_indi_list_detai) ** 2 / (num_d ** 2 + 1))+random.random()/100
                    elif (aspect_indi_list_detai[i][j]==1):
                        adj_matrix_distance_idf[i, j] = 1


            adj_matrix_distance = adj_matrix_distance_TF * adj_matrix_distance_idf
            adj_matrix_distance_detail=adj_matrix_distance.numpy()
            adj_matrix_distance_end=np.zeros((len(aspect_indi_list_detai), len(aspect_indi_list_detai[0]),len(aspect_indi_list_detai[0])))
            for k in range(len(aspect_indi_list_detai)):
                    keys_row=[]
                    for key,value in top_3(adj_matrix_distance[k]):
                        keys_row.append(key)
                    keys_col=map_AS[k].repeat(len(keys_row))
                    adj_matrix_distance_end[k,keys_row,keys_col]=1
                    adj_matrix_distance_end[k,keys_col,keys_row]=1

            #aspect-aspect方面与方面的关系
            choice_list = [(idx, idx + 1) for idx in range(len(d['aspects']) - 1)] if args.is_filtered else list(
                itertools.combinations(list(range(len(d['aspects']))), 2))
            aa_choice_inner_bert_id_list = []
            num_aspects = len(d['aspects'])
            cnum = num_aspects + len(choice_list) #方面集的数量和方面与方面对比的数量之和
            aa_graph = np.zeros((cnum, cnum))
            cnt = 0
            for aa_info in d['aa_choice']:
                select_ = (aa_info['select_idx'][0], aa_info['select_idx'][1])
                if select_ in choice_list: # choicen
                    first = aa_info['select_idx'][0] #取first的数值
                    second = aa_info['select_idx'][1] #取second的数值
                    word_range = aa_info['word_range']
                    select_words = d['token'][word_range[0]:word_range[-1] + 1] if (word_range[0] <= word_range[-1]) else ['and']
                    aa_raw_bert_ids, _, _ = text2bert_id(select_words, tokenizer)
                    aa_raw_bert_ids = aa_raw_bert_ids[:max_len]
                    if args.aa_graph_version == 1: #directional
                        if first % 2 == 0:
                            aa_graph[cnt + num_aspects][first] = 1 
                            aa_graph[second][cnt + num_aspects] = 1
                        else:
                            aa_graph[cnt + num_aspects][second] = 1
                            aa_graph[first][cnt + num_aspects] = 1
                    else: # undirectional
                        aa_graph[cnt + num_aspects][first] = 1
                        aa_graph[second][cnt + num_aspects] = 1
                    if args.aa_graph_self:
                        aa_graph[first][first] = 1 
                        aa_graph[second][second] = 1
                        aa_graph[cnt + num_aspects][cnt + num_aspects] = 1
                    aa_choice_inner_bert_id_list.append(CLS_id + aa_raw_bert_ids + SEP_id)
                    cnt += 1
            processed += [
                (
                    length, bert_length, word_mapback,
                    adj_i_oneshot, aa_graph,adj_matrix_distance_end,adj_matrix_distance_detail,
                    # aspect-specific
                    bert_sequence_list, bert_segments_ids_list, aspect_indi_list, select_spans_list,
                    # aspect-aspect
                    aa_choice_inner_bert_id_list, 
                    # label
                    label_list
                )
            ]
        return processed



def ABSA_collate_fn(batch):
    batch_size = len(batch) #batch_size的大小
    batch = list(zip(*batch))
    lens = batch[0]
    (length_, bert_length_, word_mapback_,#word_mapback代表bert将句子分割
    adj_i_oneshot_, aa_graph_,adj_matrix_distance_,adj_matrix_distance_detail_,
    bert_sequence_list_, bert_segments_ids_list_, #bert_sequence_list_ 是每个句子的bert编号组成 bert_segments_ids_list_ 是这个组成中aspect所在的位置 如[101, 2295, 1996, 2326, 2003, 26203, 2022, 4810, 2000, 3524, 2012, 2560, 2322, 2781, 2005, 1996, 11350, 15653, 11227, 1012, 102, 2326, 102] 中的[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    aspect_indi_list_, select_spans_list_, #代表aspect在句子中的位置，其中[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] select_spans_list_代表3层结构图
    aa_choice_inner_bert_id_list_, #表示句子中两个方面词在这句话中的中间关键词 [101, 2003, 26203, 2022, 4810, 2000, 3524, 2012, 2560, 2322, 2781, 2005, 1996, 102]
    label_list_) = batch
    max_lens = max(lens) #最大长度
    length = torch.LongTensor(length_)
    bert_length = torch.LongTensor(bert_length_)
    word_mapback = get_long_tensor(word_mapback_, batch_size)#batch_size个batch对应图
    adj_oneshot = np.zeros((batch_size, max_lens, max_lens), dtype=np.float32) #adj_oneshot 表示依赖树想关联的矩阵，如果存在节点存在依赖书则为1
    for idx in range(batch_size):
        mlen = adj_i_oneshot_[idx].shape[0]
        adj_oneshot[idx,:mlen,:mlen] = adj_i_oneshot_[idx]
    adj_oneshot = torch.FloatTensor(adj_oneshot)
    # Intra-context   #定义为第一项有2个 第二项有3个 00  111  序号为0 1 0 1 2
    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list_)]  #代表每一个bert_squence_list中 [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4, 4], [5, 5], [6, 6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]]每个list中的序号【0，0,1,1，1,2,2,2,3】
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list_] #[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4, 4], [5, 5], [6, 6, 6], [7, 7], [8, 8], [9, 9], [10, 10], [11, 11]] 化为[range(0, 2), range(0, 2), range(0, 2), range(0, 2), range(0, 3), range(0, 2), range(0, 3), range(0, 2), range(0, 2), range(0, 2), range(0, 2), range(0, 2)]
    # add_pre = np.array([0] + [len(m) for m in map_AS[:-1]]).cumsum()
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list])
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list]) #代表每一个bert_squence_list
    as_batch_size = len(map_AS) # 定义为长度为26
#列表展开式
    bert_sequence = [p for p_list in bert_sequence_list_ for p in p_list] #bert_sequence_list_ 展开一个一个的列
    bert_sequence = get_long_tensor(bert_sequence, as_batch_size)
#列表展开式

    # adj_matrix_distance = np.zeros((batch_size,  max_lens), dtype=np.float32) #adj_oneshot 表示依赖树想关联的矩阵，如果存在节点存在依赖书则为1
    # for idx in range(batch_size):
    #     mlen_1 = adj_matrix_distance_[idx].shape[0]
    #     mlen_2 = adj_matrix_distance_[idx].shape[1]
    #     adj_matrix_distance[idx, :mlen_1,:mlen_2] = adj_matrix_distance_[idx]
    # adj_matrix_distance = torch.FloatTensor(adj_matrix_distance)
    # #
    adj_matrix_distance=[p for p_list in adj_matrix_distance_ for p in p_list]
    adj_matrix_distance=get_float_tensor_3dim(adj_matrix_distance,as_batch_size)
    adj_matrix_distance_detail=[p for p_list in adj_matrix_distance_detail_ for p in p_list]
    adj_matrix_distance_detail=get_float_tensor(adj_matrix_distance_detail,as_batch_size)
    # adj_matrix_distance=[p for p_list in adj_matrix_distance_ for p in p_list]
    # adj_matrix_distance=get_float_tensor_3dim(adj_matrix_distance,as_batch_size)
    bert_segments_ids = [p for p_list in bert_segments_ids_list_ for p in p_list] #bert_sequence_list_ tuple(12) 展开一个一个的列 list(26)
    bert_segments_ids = get_long_tensor(bert_segments_ids, as_batch_size)
    aspect_indi = [p for p_list in aspect_indi_list_ for p in p_list]
    aspect_indi = get_long_tensor(aspect_indi, as_batch_size)
    con_spans_list = [p for p_list in select_spans_list_ for p in p_list]
    max_num_spans = max([len(p) for p in con_spans_list])
    #内容跨度图
    con_spans = np.zeros((as_batch_size, max_num_spans, max_lens), dtype=np.int64)
    for idx in range(as_batch_size):
        mlen = len(con_spans_list[idx][0])
        con_spans[idx,:,:mlen] = con_spans_list[idx]
    con_spans = torch.LongTensor(con_spans) # batch*层数*最长长度
    label = torch.LongTensor([sl for sl_list in label_list_ for sl in sl_list if isinstance(sl, int)]) #将label的尺寸平铺开
    # aa_graph 构建方面矩阵
    aspect_num = [len(a_i) for a_i in bert_sequence_list_] #有几个方面 每个句子[2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2]
    max_aspect_num = max(aspect_num) #最大的方面数量
    if (max_aspect_num > 1):
        aa_graph_length = torch.LongTensor([2 * num - 1 for num in aspect_num])  # 相当于length 2*num-1
        aa_graph = np.zeros((batch_size, 2 * max_aspect_num - 1, 2 * max_aspect_num - 1)) #构建方面矩阵
        for idx in range(batch_size):
            cnum = aa_graph_length[idx]
            aa_graph[idx, :cnum, :cnum] = aa_graph_[idx]
        aa_graph = torch.LongTensor(aa_graph) #一个句子里面方面集与方面集的关系矩阵
    else:
        aa_graph_length = torch.LongTensor([])
        aa_graph = torch.LongTensor([])
    aa_choice = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list]#aa_choice_inner_bert_id_list_ 假如有3个方面 则有2个list，第一个是第一个方面与第二个方面之间的文字，第二个为第二个方面跟第三个方面的文字 为aa
    aa_batch_size = len(aa_choice)
    if aa_batch_size > 0:
        map_AA = [[idx] * len(a_i) for idx, a_i in enumerate(aa_choice_inner_bert_id_list_)] #mapAA为反应aa_choice_inner_bert_id_list_的序列编号，如【0，1，2，3，3，3】
        map_AA = torch.LongTensor([m for m_list in map_AA for m in m_list])
        map_AA_idx = torch.LongTensor([m + len(a_i) + 1 for a_i in aa_choice_inner_bert_id_list_ for m in range(len(a_i))])  #【1，1，1，1，2】经过变换为【2，2，2，2，3，4】
        aa_choice_inner_bert_id = [m for m_list in aa_choice_inner_bert_id_list_ for m in m_list if len(m) > 0]
        aa_choice_inner_bert_length = torch.LongTensor([len(m) - 2 for m in aa_choice_inner_bert_id])
        aa_choice_inner_bert_id = get_long_tensor(aa_choice_inner_bert_id, aa_batch_size)
    else:
        map_AA = torch.LongTensor([])
        map_AA_idx = torch.LongTensor([])
        aa_choice_inner_bert_id = torch.LongTensor([])
        aa_choice_inner_bert_length = torch.LongTensor([])
    return (
        length, bert_length, word_mapback, adj_oneshot,
        aa_graph, aa_graph_length,adj_matrix_distance,adj_matrix_distance_detail,
        map_AS, map_AS_idx,
        bert_sequence, bert_segments_ids,
        aspect_indi, con_spans,
        map_AA, map_AA_idx,
        aa_choice_inner_bert_id, aa_choice_inner_bert_length,
        label
    )
def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id ,word_mapback, word_split_len


class ABSA_DataLoader(DataLoader):
    def __init__(self, dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        '''
        :param dataset: Dataset object
        :param sort_idx: sort_function
        :param sort_bs_num: sort range; default is None(sort for all sequence)
        :param is_shuffle: shuffle chunk , default if True
        :return:
        '''
        assert isinstance(dataset.data, list)
        super().__init__(dataset,**kwargs)
        self.sort_key = sort_key
        self.sort_bs_num = sort_bs_num
        self.is_shuffle = is_shuffle

    def __iter__(self):
        if self.is_shuffle:
            self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num, self.sort_key, self.is_shuffle)

        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # sort
        random.shuffle(data)
        data = sorted(data, key = sort_key) # 先按照长度排序
        batch_data = [data[i : i + batch_size] for i in range(0,len(data),batch_size)]
        batch_data = [sorted(batch, key = sort_key) for batch in batch_data]
        if is_shuffle:
            random.shuffle(batch_data)
        batch_data = list(chain(*batch_data))
        return batch_data

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens
def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens
def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]
def get_float_tensor_3dim(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    x_max=0
    y_max=0
    for i in range(len(tokens_list)):
        token_len_x = max(len(x) for x in tokens_list[i])
        token_len_y = max(len(y) for y in tokens_list[i])
        if(x_max<token_len_x):
            x_max=token_len_x
        if(y_max<token_len_y):
            y_max=token_len_y
    tokens = torch.FloatTensor(batch_size, x_max,y_max).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s),: len(s)] = torch.FloatTensor(s)
    return tokens
def index_found(a,k):
    index_aa=[]
    for index,value in enumerate(a):
        if(value>=k):
            index_aa.append(index)
    return index_aa

def top_3(nums):
    return heapq.nlargest(8, enumerate(nums), key=lambda x: x[1])
