from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer
from tqdm.auto import tqdm
import torch

MODEL_NAMES = [
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_t5_xl_bfd",
    "Rostlab/prot_t5_xxl_uniref50",
    "Rostlab/prot_t5_xxl_bfd",
    "Rostlab/prot_bert_bfd",
    "Rostlab/prot_bert",
    "Rostlab/prot_xlnet",
    "Rostlab/prot_albert",
]


def get_pre_model(model_name, device):
    pre_model, tokenizer = get_model_and_tokenizer(model_name)
    shift_left, shift_right = get_padding_shifts(model_name)
    pre_model = pre_model.to(device)
    pre_model = pre_model.eval()
    if torch.cuda.is_available():
        pre_model = pre_model.half()
    return pre_model, tokenizer, shift_left, shift_right


def get_model_and_tokenizer(model_name):
    if model_name not in MODEL_NAMES:
        raise Exception("Unknown model")

    if "t5" in model_name:
        pre_tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        pre_model = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        pre_tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        pre_model = AlbertModel.from_pretrained(model_name)
    elif "bert" in model_name:
        pre_tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        pre_model = BertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        pre_tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False)
        pre_model = XLNetModel.from_pretrained(model_name)
    else:
        raise Exception("Unknown model")

    return pre_model, pre_tokenizer


def get_padding_shifts(model_name):
    if model_name not in MODEL_NAMES:
        raise Exception("Unknown model")

    if "t5" in model_name:
        shift_left = 0
        shift_right = -1
    elif "bert" in model_name:
        shift_left = 1
        shift_right = -1
    elif "xlnet" in model_name:
        shift_left = 0
        shift_right = -2
    elif "albert" in model_name:
        shift_left = 1
        shift_right = -1
    else:
        raise Exception("Unknown model")

    return shift_left, shift_right


def get_embedding(
    dataset_seqs, shift_left, shift_right, pre_model, pre_tokenizer, device
):
    inputs_embedding = []

    for sample in tqdm(dataset_seqs):
        with torch.no_grad():
            # FIX: batch_encode_plus removed in transformers 5.x; use tokenizer directly
            ids = pre_tokenizer(
                [sample],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = pre_model(input_ids=ids["input_ids"].to(device))[0]
            inputs_embedding.append(
                embedding[0].detach().cpu().numpy()[shift_left:shift_right]
            )

    return inputs_embedding
