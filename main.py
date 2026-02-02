from networkx import from_prufer_sequence, tadpole_graph
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        QK = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            QK = QK.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(QK / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        Z = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(Z)
        X = self.dropout(self.norm2(forward + Z))

        return X
    
class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
    ):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.Module(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                )
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(self.word_embedding(x) + self.positional_embedding(pos))

        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()

        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, src_mask)

        return out
    
class Decoder(nn.Module):
    def __init__(
            self,
            target_vocab_size,
            embed_size,
            num_layers,
            heads, 
            forward_expansion,
            dropout,
            device,
            max_length
    ):
        super(Decoder, self).__init__()

        self.device = device
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, target_mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        
        x = self.dropout((self.word_embedding(x) + self.positional_embedding(pos)))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)

        out = self.fc_out(x)