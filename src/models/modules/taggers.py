import torch
from torch import nn
from src.models.modules.transformers import TransformerBlock


class LSTMTagger(nn.Module):
    def __init__(
        self, pre_tokenizer, pre_model, shift_left, shift_right, device, deep=False
    ):
        super(LSTMTagger, self).__init__()

        self.device = device
        self.pre_tokenizer = pre_tokenizer
        self.pre_model = pre_model
        self.shift_left = shift_left
        self.shift_right = shift_right

        if not deep:
            self.net = nn.Sequential(
                nn.LazyLinear(512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU()
            )

            self.lstm = nn.LSTM(256, 128, bidirectional=True)

            self.hidden2tag = nn.Sequential(
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid()
            )
        else:
            self.net = nn.Sequential(
                nn.LazyLinear(1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
            )

            self.lstm = nn.LSTM(128, 64, bidirectional=True)

            self.hidden2tag = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, sequence):
        seq = " ".join(sequence).split()
        with torch.no_grad():
            # FIX: batch_encode_plus removed in transformers 5.x; use tokenizer directly
            ids = self.pre_tokenizer(
                [seq],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = self.pre_model(input_ids=ids["input_ids"].to(self.device))[0]
            embeds = embedding[0].detach()[self.shift_left : self.shift_right]
            embeds = embeds.float()

        net = self.net(embeds)
        lstm_out, _ = self.lstm(net.view(len(sequence), 1, -1))
        tag_score = self.hidden2tag(lstm_out.view(len(sequence), -1))
        return tag_score


class TransformerLSTMTagger(nn.Module):
    def __init__(
        self,
        pre_tokenizer,
        pre_model,
        shift_left,
        shift_right,
        device,
        input_dim,
        linear_hidden_dim,
        num_heads,
        num_blocks,
        dropout,
        with_lstm=False,
        lstm_n_layers=2,
    ):
        super(TransformerLSTMTagger, self).__init__()

        self.device = device
        self.pre_tokenizer = pre_tokenizer
        self.pre_model = pre_model
        self.shift_left = shift_left
        self.shift_right = shift_right
        self.with_lstm = with_lstm

        self.en_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.en_blocks.add_module(
                "block" + str(i),
                TransformerBlock(input_dim, linear_hidden_dim, num_heads, dropout),
            )

        self.lstm = None
        if self.with_lstm:
            self.lstm = nn.LSTM(
                input_dim,
                256,
                bidirectional=True,
                batch_first=True,
                num_layers=lstm_n_layers,
                dropout=dropout,
            )

        self.hidden2tag = nn.Sequential(
            nn.Linear(512 if self.with_lstm else input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, sequences):
        seqs = []
        for sequence in sequences:
            seq = " ".join(sequence).split()
            seqs.append(seq)

        with torch.no_grad():
            # FIX: batch_encode_plus removed in transformers 5.x; use tokenizer directly
            ids = self.pre_tokenizer(
                seqs,
                add_special_tokens=False,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embeds = self.pre_model(input_ids=ids["input_ids"].to(self.device))[
                0
            ].float()

        X = embeds
        for i, blk in enumerate(self.en_blocks):
            X = blk(X)

        if self.with_lstm:
            X, _ = self.lstm(X)

        return self.hidden2tag(X)
