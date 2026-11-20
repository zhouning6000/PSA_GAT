import torch
import torch.nn as nn
import warnings
import torch.nn.functional as F
import copy

class H_GCNEncoder(nn.Module):
    '''
    Transformer Encoder
    '''

    def __init__(self, d_model=200,
                 num_encoder_layers=3 ,
                 inner_encoder_layers = 5,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation='relu',
                 layer_norm_eps=1e-5):

        super(H_GCNEncoder, self).__init__()
        encoder_layer = H_GCNEncoderlayer(d_model=d_model,#512
                                                inner_layer=inner_encoder_layers,#3
                                                dim_feedforward=dim_feedforward,#2048
                                                dropout=dropout,#0.1
                                                activation=activation,#
                                                layer_norm_eps=layer_norm_eps)

        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.layers = _get_clones(encoder_layer, num_encoder_layers)  #复制3层
        self.num_layers = num_encoder_layers
        self._reset_parameters()

    def _reset_parameters(self):
        '''
        Initiate parameters in the transformer model
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask): #[表示相同操作重复num_encoder_layer次]
        B, L, _ = src.shape
        output = src  # src:[S,bs,E]

        for layer in self.layers:
            output = layer(output, mask)
        if self.encoder_norm is not None:
            output = self.encoder_norm(output)
        return output  # [bs,S,E]
class H_GCNEncoderlayer(nn.Module):
    def __init__(self, d_model, inner_layer = 3, dim_feedforward=2048, dropout=0.1, activation='relu',layer_norm_eps=1e-5):
        super(H_GCNEncoderlayer, self).__init__()
        inner_encoder_layer = TransformerGCNEncoderLayer(d_model=d_model,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                activation=activation,
                                                layer_norm_eps=layer_norm_eps)
        self.layers = _get_clones(inner_encoder_layer, inner_layer)#复制3遍
        self.encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, src, src_mask):
        output = src
        L,B,D = src.shape #长度
        for idx, layer in enumerate(self.layers):
            output = layer(output, src_mask=src_mask)
            assert(torch.isnan(output).sum() == 0)
        if self.encoder_norm is not None:
            output = self.encoder_norm(output)
            return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerGCNEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5):
        super(TransformerGCNEncoderLayer, self).__init__()
        self.GCN = GCN(d_model, d_model,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.feedforword = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            _get_activation_fn(activation),
            self.dropout,
            nn.Linear(dim_feedforward, d_model),
            self.dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
    def forward(self, src, src_mask=None):
        src2 = self.GCN(src, src_mask)
        src_add_norm = self.norm(src + self.dropout(src2))
        return self.norm(src + self.feedforword(src_add_norm))
def _get_activation_fn(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))



class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout):
        super(GCN, self).__init__()
        self.first_layer = GraphConvolution(n_features, dropout=0.1)
        self.second_layer = GraphConvolution(n_features, dropout=0.1)
        self.third_layer = GraphConvolution(n_features, dropout=0.1)
        self.last_layer = GraphConvolution(hidden_dim, dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    def forward(self, inputs, adj):
        x = self.relu(self.first_layer(inputs, adj))
        x = self.dropout(x)
        x = self.relu(self.second_layer(inputs, adj))
        x = self.dropout(x)
        x = self.relu(self.third_layer(inputs, adj))
        x = self.dropout(x)
        x = self.last_layer(x, adj)
        return F.log_softmax(x, dim=1)

class GraphConvolution(nn.Module):
    def __init__(self, features,dropout):
        super(GraphConvolution, self).__init__()
        self.in_features = features
        self.out_features = features
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Linear(features, features, bias=True)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.w.weight)
    def forward(self, input, adj):
        # inputs: (batch,N, n_channels), ->support(batch N out_feats)
        support = self.w(self.dropout(input))
        # output adj: sparse_matrix (batch,N, N) support(batch,N,out_feats) ->output(batch,N,OUT_feats)
        output = torch.bmm(adj, support)
        return output