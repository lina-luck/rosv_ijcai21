import torch.nn as nn
import torch

class BaseRGCN(nn.Module):
    def __init__(self, in_feat, h_dim, out_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, features=None):
        super(BaseRGCN, self).__init__()
        self.in_feat = in_feat
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_cuda = use_cuda
        self.features = features

        # create rgcn layers
        self.build_model()

        # create initial features
        if self.features is None:
            self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)

        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        return None

    def build_input_layer(self):
        return None

    def build_hidden_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g):
        if self.use_cuda:
            g.ndata['id'] = torch.arange(len(g)).to('cuda')
        else:
            g.ndata['id'] = torch.arange(len(g))
        for layer in self.layers:
            layer(g)
        return g.ndata.pop('h')
