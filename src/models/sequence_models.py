from base.basemodel import BaseModel
from base.data import Data
from src.models.pre_trained_helper import get_pre_model
from src.models.modules.taggers import TransformerLSTMTagger
from torch import nn


class TransformerLSTMModel(BaseModel):

    def __init__(
        self,
        pre_model_name,
        device,
        input_dim,
        linear_hidden_dim,
        num_heads,
        num_blocks,
        dropout,
        model_config,
        with_lstm=False,
        lstm_n_layers=2,
    ):
        super().__init__(model_config)
        pre_model, pre_tokenizer, shift_left, shift_right = get_pre_model(
            model_name=pre_model_name, device=device
        )
        self.model = TransformerLSTMTagger(
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
            with_lstm,
            lstm_n_layers,
        ).to(device)

    def destroy(self):
        pass

    def predict(self, X):
        pred = self.model(X)
        pred = nn.Softmax(dim=2)(pred)
        return pred.argmax(2).squeeze().tolist()

    def summary(self):
        pass

    def evaluate(self, data: Data, thresh):
        pass

    def load(self):
        m = 'LongTagger_' + model_name[8:] + '.pth'
        self.model.load_state_dict(torch.load(m))
