from torch import nn


class PositionWiseFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class TransformerBlock(nn.Module):
    def __init__(
        self, input_dim, linear_hidden_dim, num_heads, dropout, use_bias=False
    ):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            input_dim, num_heads, dropout, use_bias, batch_first=True
        )
        self.add_norm1 = AddNorm(input_dim, dropout)
        self.ffn = PositionWiseFFN(input_dim, linear_hidden_dim, input_dim)
        self.add_norm2 = AddNorm(input_dim, dropout)

    def forward(self, X):
        Y = self.add_norm1(X, self.attention(X, X, X)[0])
        return self.add_norm2(Y, self.ffn(Y))
