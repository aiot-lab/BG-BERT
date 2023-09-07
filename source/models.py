import math
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."
    def __init__(self, variance_epsilon=1e-12, **kwargs):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(kwargs['hidden']), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(kwargs['hidden']), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class Embeddings(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.lin = nn.Linear(kwargs['feature_num'], kwargs['hidden'])
        self.pos_embed = nn.Embedding(kwargs['seq_len'], kwargs['hidden']) # position embedding

        self.norm = LayerNorm(**kwargs)
        self.emb_norm = kwargs['emb_norm']

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)

        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)

class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, **kwargs):
        super().__init__()
        self.proj_q = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.proj_k = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.proj_v = nn.Linear(kwargs['hidden'], kwargs['hidden'])

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, **kwargs):
        super().__init__()
        self.proj_q = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.proj_k = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.proj_v = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.scores = None
        self.n_heads = kwargs['n_heads']

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(kwargs['hidden'], kwargs['hidden_ff'])
        self.fc2 = nn.Linear(kwargs['hidden_ff'], kwargs['hidden'])

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))

class ResidualConnectionModule(nn.Module):
    """
    Residual Connection Module.
    outputs = (module(inputs) x module_factor + inputs x input_factor)
    """
    def __init__(self, module: nn.Module, module_factor: float = 1.0, input_factor: float = 1.0):
        super(ResidualConnectionModule, self).__init__()
        self.module = module
        self.module_factor = module_factor
        self.input_factor = input_factor

    def forward(self, inputs: Tensor) -> Tensor:
        return (self.module(inputs) * self.module_factor) + (inputs * self.input_factor)

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, **kwargs):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(**kwargs)
        self.proj = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.norm1 = LayerNorm(**kwargs)
        self.pwff = PositionWiseFeedForward(**kwargs)
        self.norm2 = LayerNorm(**kwargs)

    def forward(self, x):
        h = self.attn(x)
        h = self.norm1(h + self.proj(h))
        h = self.norm2(h + self.pwff(h))
        return h

class Transformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.embed = Embeddings(**kwargs)
        self.n_layers = kwargs['n_layers']
        self.dropout = kwargs['dropout']
        self.sequentials = nn.ModuleList(
            [ResidualConnectionModule(TransformerBlock(**kwargs),
                                       module_factor=1.0, input_factor=1.0) for _ in range(self.n_layers)]
        )

    def forward(self, x):
        h = self.embed(x)
        for layer in self.sequentials:
            h = layer(h)
        return h

class TransformerSameParameter(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, **kwargs):
        super().__init__()
        self.embed = Embeddings(**kwargs)
        self.n_layers = kwargs['n_layers']
        self.attn = MultiHeadedSelfAttention(**kwargs)
        self.proj = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.norm1 = LayerNorm(**kwargs)
        self.pwff = PositionWiseFeedForward(**kwargs)
        self.norm2 = LayerNorm(**kwargs)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h

class BGBertClassification(nn.Module):
    def __init__(self, **kwargs):
        super(BGBertClassification, self).__init__()
        self.conv1 = nn.Conv1d(kwargs['seq_len'], kwargs['fore_length'], kernel_size=1, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(kwargs['hidden'], kwargs['hidden'] // 2)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(kwargs['hidden'] // 2, kwargs['class_num'])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        # average pooling
        x = torch.mean(x, dim=1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = x.squeeze(-1)
        x = self.softmax(x)
        return x

class BGBertModel4Pretrain(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.transformer = Transformer(**kwargs) # encoder
        self.fc = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.linear = nn.Linear(kwargs['hidden'], kwargs['hidden'])
        self.activ = gelu
        self.norm = LayerNorm(**kwargs)
        self.decoder = nn.Linear(kwargs['hidden'], kwargs['feature_num'])
        self.output_embed = kwargs['output_embed']

    def forward(self, input_seqs, masked_pos=None):
        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        h_masked = self.decoder(h_masked)

        return h_masked

class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)

class Swish(nn.Module):
    """
    Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks applied
    to a variety of challenging domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * inputs.sigmoid()

class BilinearLSTMSeqNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        """
        LSTM network with Bilinear layer
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]
        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(BilinearLSTMSeqNetwork, self).__init__()
        self.input_size = kwargs['hidden']
        self.lstm_size = kwargs['hidden']
        self.output_size = kwargs['hidden']
        self.num_layers = 3
        self.batch_size = kwargs['batch_size']
        self.device = kwargs['device']

        self.bilinear = torch.nn.Bilinear(self.input_size, self.input_size, self.input_size * 4)
        self.lstm = torch.nn.LSTM(self.input_size * 5, self.lstm_size, self.num_layers, batch_first=True)
        self.linear1 = torch.nn.Linear(self.lstm_size + self.input_size * 5, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size * 5, self.output_size)
        self.hidden = self.init_weights()

    def forward(self, input):
        input_mix = self.bilinear(input, input)
        input_mix = torch.cat([input, input_mix], dim=2)
        output, _ = self.lstm(input_mix)
        output = torch.cat([input_mix, output], dim=2)
        output = self.linear1(output)
        output = self.linear2(output)
        return output

    def init_weights(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.lstm_size)
        h0 = h0.to(self.device)
        c0 = c0.to(self.device)
        return Variable(h0), Variable(c0)

class BGBertPrediction(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.rnn = BilinearLSTMSeqNetwork(**kwargs)
        self.projection = nn.Sequential(nn.Conv1d((kwargs['seq_len']), kwargs['fore_length'] * 3, kernel_size=1, stride=1, padding=0),
                                        nn.ReLU(),
                                        nn.Conv1d(kwargs['fore_length'] * 3, kwargs['fore_length'], kernel_size=1, stride=1, padding=0),
                                        nn.ReLU())
        self.feedforward = nn.Sequential(nn.LayerNorm(kwargs['hidden']),
                                         Linear(kwargs['hidden'], kwargs['hidden'] * 2, bias=True),
                                         Swish(),
                                         Linear(kwargs['hidden'] * 2, kwargs['hidden'], bias=True),
                                         Swish(),
                                         Linear(kwargs['hidden'], kwargs['hidden'] // 2, bias=True),
                                         Swish(),
                                         Linear(kwargs['hidden'] // 2, 1, bias=True),
                                         nn.ReLU())

    def forward(self, input_representation):
        x = input_representation
        x = self.rnn(x)
        x = self.projection(x)
        x = self.feedforward(x)
        x = x.squeeze(-1)
        return x
