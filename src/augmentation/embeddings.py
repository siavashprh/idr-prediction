import torch
from pathlib import Path
from tqdm.auto import tqdm

from src.data.data import get_disprot_cut_data


def extract_embeddings(
    pre_model,
    pre_tokenizer,
    device: str,
    save_path: str,
) -> list[torch.Tensor]:
    """Extract per-residue ProtBERT embeddings for all DisProt training chunks.

    Uses add_special_tokens=False to match TransformerLSTMTagger's embedding
    convention (no CLS/SEP tokens, no index shifting needed).

    Args:
        pre_model: Frozen ProtBERT BertModel on `device`.
        pre_tokenizer: BertTokenizer for prot_bert_bfd.
        device: "cuda" or "cpu".
        save_path: Path to write the list of tensors (torch.save format).

    Returns:
        List of float32 tensors of shape (seq_len, 1024), one per training chunk.
    """
    cut = get_disprot_cut_data()
    sequences = cut["sequences"]
    embeddings: list[torch.Tensor] = []

    pre_model.eval()
    for seq in tqdm(sequences, desc="Extracting embeddings"):
        with torch.no_grad():
            ids = pre_tokenizer(
                [list(seq)],
                add_special_tokens=False,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            emb = pre_model(input_ids=ids["input_ids"].to(device))[0]
            embeddings.append(emb[0].float().cpu())  # (seq_len, 1024)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, save_path)
    return embeddings
