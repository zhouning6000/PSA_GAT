import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from layer import H_TransformerEncoder


class BiSyn_GAT_plus(nn.Module):
    def __init__(self, args):
        super().__init__()
        in_dim = args.bert_hidden_dim + args.hidden_dim
        self.args = args

        self.intra_context_module = Intra_context(args)

        # inter_context_module
        if args.plus_AA:
            self.inter_context_module = Inter_context(args)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, args.num_class)

    def forward(self, inputs):
        length, bert_length, word_mapback, \
        adj, aa_graph, aa_graph_length, adj_matrix_distance,\
        map_AS, map_AS_idx, \
        bert_sequence, bert_segments_ids, \
        aspect_indi, \
        con_spans, \
        map_AA, map_AA_idx, \
        aa_choice_inner_bert, aa_choice_inner_bert_length = inputs

        # aspect-level
        #输入到intra_context_input的数据如下
        Intra_context_input = (
        length[map_AS], bert_length[map_AS], word_mapback[map_AS], aspect_indi, adj_matrix_distance,bert_sequence, bert_segments_ids,
        adj[map_AS], con_spans)
# 输入 长度 字符对照表 方面词对照 bert序列 bert标志
        Intra_context_output = self.intra_context_module(Intra_context_input)
#将每句话的向量缩成一个200维度的向量 同时添加上整个句子的bertout 输出一个特征值
        # sentence-level
        if self.args.plus_AA and map_AA.numel():  # BiSyn-GAT+
            Inter_context_input = (aa_choice_inner_bert, aa_choice_inner_bert_length, #两个方面词之间的句子 #两个方面词相隔的距离
                                   map_AA, map_AA_idx, map_AS, map_AS_idx, #两个方面对应
                                   aa_graph_length, aa_graph)
            # sentence-level to aspect-level
            hiddens = self.inter_context_module(as_features=Intra_context_output,
                                                inputs=Inter_context_input, \
                                                context_encoder=self.intra_context_module.context_encoder if self.args.borrow_encoder else None)

        else:  # BiSyn-GAT
            hiddens = Intra_context_output

            # aspect-level
        logits = self.classifier(self.dropout(hiddens))
        return logits


# Intra-context module
class Intra_context(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        bert_config = BertConfig.from_pretrained('data/bert-base-uncased')
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3

        self.layer_drop = nn.Dropout(args.layer_dropout)
        self.context_encoder = BertModel.from_pretrained('data/bert-base-uncased', config=bert_config)
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)
        self.graph_encoder = H_TransformerEncoder(
            d_model=args.hidden_dim,
            nhead=args.attn_head,
            num_encoder_layers=args.num_encoder_layer,
            inner_encoder_layers=args.max_num_spans,
            dropout=args.layer_dropout,
            dim_feedforward=args.bert_hidden_dim,
            activation='relu',
            layer_norm_eps=1e-5
        )

    def forward(self, inputs):

        length, bert_lengths, word_mapback, mask,adj_matrix_distance, bert_sequence, bert_segments_ids, adj, con_spans = inputs

        ############################################################### 输入bert编码
        # 1. contextual encoder
        bert_outputs = self.context_encoder(bert_sequence, token_type_ids=bert_segments_ids)

        bert_out_8,bert_out, bert_pooler_out = bert_outputs.hidden_states[8], bert_outputs.last_hidden_state, bert_outputs.pooler_output
        bert_out_8= self.layer_drop(bert_out_8)
        bert_out = self.layer_drop(bert_out)
       #
        # rm [CLS]
        bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float() #对bertout中结尾部分进行截取操作 变成【0.0，0.0.0.0.0.0.0.0】
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)  #因为F.one_hot(word_mapback)中本来为0的数会变为【1，0，0，0。。。。】  经过这一步的操作结尾为【1，0，0，0，】的变为【0，0，0，0，0，0。0，】 同时将 batch*句子 变成 batch*[二维矩阵] 同时 变换为batch*维度*句子长度
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out)) # batch*维度*每个句子单词数  batch* 每个句子单词数 * 特征值=  batch
        bert_out_8 = bert_out_8[:, 1:max(bert_lengths) + 1,
                     :] * bert_seq_indi.float()  # 对bertout中结尾部分进行截取操作 变成【0.0，0.0.0.0.0.0.0.0】
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1,
                                                                                                   2)  # 因为F.one_hot(word_mapback)中本来为0的数会变为【1，0，0，0。。。。】  经过这一步的操作结尾为【1，0，0，0，】的变为【0，0，0，0，0，0。0，】 同时将 batch*句子 变成 batch*[二维矩阵] 同时 变换为batch*维度*句子长度
        bert_out_8 = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out_8))

        #bert_out=bert_out*adj_matrix_distance.unsqueeze(-1)
        # average 输入均值
        wnt = word_mapback_one_hot.sum(dim=-1)#代表每个词在句子中分解的数量
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)
        bert_out_8 = bert_out_8 / wnt.unsqueeze(dim=-1)

        ###############################################################
        # 2. graph encoder 图编码器
        key_padding_mask = sequence_mask(length)  # [B, seq_len]

        # from phrase(span) to form mask 语法编码
        B, N, L = con_spans.shape #Batch N是句子层 L是句子中的长度
        span_matrix = get_span_matrix_4D(con_spans.transpose(0, 1))

        if self.args.con_dep_version == 'con_add_dep':
            # adj + span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1)
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0)
            span_matrix = (span_matrix + adj_matrix).bool()

        elif self.args.con_dep_version == 'wo_dep':
            # only span #只进行依赖编码如下
            pass

        elif self.args.con_dep_version == 'wo_con':
            # only adj #只进行结构树编码
            adj_matrix = adj.unsqueeze(dim=0)
            span_matrix = adj_matrix.repeat(N, 1, 1, 1)

        elif self.args.con_dep_version == 'con_dot_dep':
            # adj * span
            adj_matrix = adj.unsqueeze(dim=0).repeat(N, 1, 1, 1)  #将依赖树 复制为N层，为了跟结构树相互对应
            assert ((adj_matrix[0] != adj_matrix[1]).sum() == 0)#判断是否一致
            span_matrix = (span_matrix * adj_matrix).bool() #依赖树和结构树想融合

        #graph_out = self.graph_encoder(bert_out_8, mask=span_matrix, src_key_padding_mask=key_padding_mask)
        #*******#
        #adj_matrix_distance = F.normalize(adj_matrix_distance, p=2, dim=1)
        #graph_out=graph_out*adj_matrix_distance.unsqueeze(-1)

        ###############################################################
        # 3. fusion
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h 为位置标注向量集合
        # h_t
        bert_enc_outputs = (bert_out * mask).sum(dim=1) / asp_wn #每句话压缩成一个向量，例如 Batch*sq*feature 压缩成Batch*feature

        # g_t
        graph_enc_outputs = (bert_out_8 * mask).sum(dim=1) / asp_wn  # mask h#每句话压缩成一个向量，例如 Batch*sq*feature 压缩成Batch*feature

        as_features = torch.cat([bert_enc_outputs+graph_enc_outputs, bert_pooler_out], -1)
        return as_features


# Inter-context module
class Inter_context(nn.Module):
    def __init__(self, args, sent_encoder=None):
        super().__init__()
        self.args = args
        in_dim = args.bert_hidden_dim + args.hidden_dim
        self.layer_drop = nn.Dropout(args.layer_dropout)
        if not args.borrow_encoder:
            self.sent_encoder = BertModel.from_pretrained('data/bert-base-uncased')
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)

        self.con_aspect_graph_encoder = H_TransformerEncoder(
            d_model=in_dim,
            nhead=args.attn_head,
            num_encoder_layers=args.aspect_graph_num_layer,
            inner_encoder_layers=args.aspect_graph_encoder_version,
            dropout=args.layer_dropout,
            dim_feedforward=args.hidden_dim,
            activation='relu',
            layer_norm_eps=1e-5
        )

    def forward(self, as_features, inputs, context_encoder=None):
        aa_choice_inner_bert, aa_choice_inner_bert_length, \
        map_AA, map_AA_idx, map_AS, map_AS_idx, \
        aa_graph_length, aa_graph = inputs   #aa_choice_inner_bert 为方面集之间的连词  map_AA为aa_choice_inner_bert_id_list_的序号，表示方面集与方面集的联系 map_AS为方面集与句子之间的联系

        need_change = (aa_graph_length[map_AS] > 1) #方面集与方面集的关系矩阵

        inner_v_node, inner_v = self.forward_bert_inner(aa_choice_inner_bert,
                                                        aa_choice_inner_bert_length,
                                                        context_encoder) #对一个句子的方面集与方面集之间的内容进行编码

        rela_v_inner = torch.cat((inner_v_node.sum(dim=1), inner_v), dim=-1)  #按第一个维度合并inner_v_node 并于inner_V在最后一个维度合并（batch*200 batch*768）进行拼接
        AA_features = self.con_aspect_graph(rela_v_inner, #节点特征与句法特征融合
                                            as_features,  #上面的整句话的bertout输出的特征值
                                            map_AA, map_AA_idx, #map_AA为aa_choice_inner_bert_id_list_的序号，表示方面集与方面集的联系
                                            map_AS, map_AS_idx, #map_AS为方面集与句子之间的联系
                                            aa_graph_length, aa_graph) #组建方面集的矩阵 且通过图神经网络输出特征

        AA_features = AA_features * need_change.unsqueeze(dim=-1) + as_features * ~(need_change).unsqueeze(dim=-1) #混合矩阵

        fusion_features = AA_features + as_features

        return fusion_features

    def forward_bert_inner(self, aa_choice_inner_bert, aa_choice_inner_bert_length, context_encoder=None):

        bert_outputs = self.sent_encoder(aa_choice_inner_bert) if context_encoder is None else context_encoder(
            aa_choice_inner_bert) #方面词与方面词之间的内容进行bert编码
        bert_out, bert_pool_output = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        bert_out = self.layer_drop(bert_out)#对中间内容进行编码 并且dropout防止过拟合
        # rm [CLS] representation
        bert_seq_indi = ~sequence_mask(aa_choice_inner_bert_length).unsqueeze(dim=-1) # 对数据进行True False
        bert_out = bert_out[:, 1:max(aa_choice_inner_bert_length) + 1, :] * bert_seq_indi.float() #对bertout进行编码重组，没有内容的归为0

        inner_v_node = self.dense(bert_out) #将bert编码转为200维度的向量 形状为batch* 长度*向量维度
        inner_v = bert_pool_output #句子向量
        return inner_v_node, inner_v
# forward_bert_inner为
    def con_aspect_graph(self,
                         rela_v,
                         as_features,
                         map_AA, map_AA_idx, map_AS, map_AS_idx,
                         aa_graph_length, aa_graph):
        B = map_AS.max() + 1# 一共有多少个句子
        L = map_AA_idx.max() + 1 #每个句子最多有多少个方面
        graph_input_features = torch.zeros((B, L, as_features.shape[-1]), device=as_features.device) #构造输入的矩阵（句子的长度*句子最大的方面*特征数量）

        graph_input_features[map_AA, map_AA_idx] = rela_v   #AA之间关系特征【句子号，方面编码】=方面之间的特征值
        graph_input_features[map_AS, map_AS_idx] = as_features   #AS之间关系特征【扩充句子号，扩充句子与方面集编号】=方面集的特征

        aa_graph_key_padding_mask = sequence_mask(aa_graph_length)  #关系矩阵填充词 TTTTFF

        if self.args.aspect_graph_encoder_version == 1:#正向加逆向的融合 通过图神经网络
            # split and share parameters
            forward_ = self.con_aspect_graph_encoder(graph_input_features, #句子方面之间的最大矩阵
                                                     mask=aa_graph.unsqueeze(0),
                                                     src_key_padding_mask=aa_graph_key_padding_mask)

            backward_ = self.con_aspect_graph_encoder(graph_input_features,
                                                      mask=aa_graph.transpose(1, 2).unsqueeze(0),
                                                      src_key_padding_mask=aa_graph_key_padding_mask)
            mutual_influence = forward_ + backward_

        elif self.args.aspect_graph_encoder_version == 2:
            # not split
            mutual_influence = self.con_aspect_graph_encoder(
                graph_input_features,
                mask=torch.cat((aa_graph.unsqueeze(dim=0), aa_graph.transpose(1, 2).unsqueeze(dim=0)), dim=0),
                src_key_padding_mask=aa_graph_key_padding_mask
            )
        return mutual_influence[map_AS, map_AS_idx]


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))


def get_span_matrix_4D(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span


def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span