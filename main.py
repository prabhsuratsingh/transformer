import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size should be divisible by number of heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        QK = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            QK = QK.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(QK / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length=5000):
        super().__init__()

        pe = torch.zeros(max_length, embed_size)
        position = torch.arange(0, max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_size, 2) *
            (-math.log(10000.0) / embed_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
        self.position_encoding = PositionalEncoding(embed_size, max_length)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion
                ) for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)

        out = self.dropout(
            self.position_encoding(self.word_embedding(x))
        )

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
        self.position_encoding = PositionalEncoding(embed_size, max_length)


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
        
        x = self.dropout(
            self.position_encoding(self.word_embedding(x))
        )


        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, target_mask)

        out = self.fc_out(x)

        return out

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            target_vocab_size,
            src_pad_index,
            target_pad_index,
            embed_size=256,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            target_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_index = src_pad_index
        self.target_pad_index = target_pad_index
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_index).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            N, 1, target_len, target_len
        )

        return target_mask.to(self.device)
    
    def forward(self, src, target):
        src_mask = self.make_src_mask(src)
        target_mask = self.make_target_mask(target)

        encoder_src = self.encoder(src, src_mask)

        out = self.decoder(target, encoder_src, src_mask, target_mask)

        return out
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    target = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_index = 0
    target_pad_index = 0
    src_vocab_size = 10
    target_vocab_size = 10

    model = Transformer(src_vocab_size, target_vocab_size, src_pad_index, target_pad_index, device=device).to(device)

    out = model(x, target[:, :-1])

    print(out.shape)