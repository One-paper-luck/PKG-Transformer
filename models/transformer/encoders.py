from models.transformer.attention import MultiHeadAttention
import numpy as np
from ..GAT.GAT import *
from models.transformer.utils import *


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                isencoder=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights, input_gl=input_gl, memory=memory,
                         isencoder=isencoder)
        ff = self.pwff(att)
        return ff


class EncoderLayer_gl_lo(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=4, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer_gl_lo, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None, input_gl=None, memory=None,
                isencoder=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights, input_gl=input_gl, memory=memory,
                         isencoder=isencoder)
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.padding_idx = padding_idx

    def forward(self, input, input_gl=None, isencoder=None, attention_weights=None):
        attention_mask_lo = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)
        out_lo=input
        out_gl=input_gl

        # prior scene-object knowledge
        memory = torch.matmul(out_lo, out_gl.permute(0, 2, 1)) / np.sqrt(self.d_model)
        memory = torch.softmax(memory, -2).sum(dim=-1)


        out = out_lo
        outs = []
        for l in self.layers:
            out = l(out, out, out, attention_mask_lo, attention_weights, input_gl=out_gl, memory=memory,
                    isencoder=isencoder)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs, attention_mask_lo

class MemoryAugmentedEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoder, self).__init__(N, padding_idx, **kwargs)
        config_img_lo = GATopt(512, 1)
        config_img_gl = GATopt(512, 1)
        self.gat_1 = GAT(config_img_lo)
        self.gat_2 = GAT(config_img_gl)
        self.fc_lo = nn.Linear(1024, self.d_model)
        self.fc_gl = nn.Linear(2048, self.d_model)
        self.dropout_lo = nn.Dropout(p=self.dropout)
        self.dropout_gl = nn.Dropout(p=self.dropout)
        self.layer_norm_lo = nn.LayerNorm(self.d_model)
        self.layer_norm_gl = nn.LayerNorm(self.d_model)

    def forward(self, input, input_gl=None, isencoder=None, attention_weights=None):

        lo = F.relu(self.fc_lo(input))
        lo = self.dropout_lo(lo)
        lo = self.layer_norm_lo(lo)

        gl = F.relu(self.fc_gl(input_gl))
        gl = self.dropout_gl(gl)
        gl = self.layer_norm_gl(gl)

        # GAT
        lo = self.gat_1(lo)
        gl = self.gat_2(gl)
        return super(MemoryAugmentedEncoder, self).forward(lo, input_gl=gl, isencoder=isencoder,
                                                           attention_weights=attention_weights)

