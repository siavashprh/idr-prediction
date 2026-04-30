"""Embedding decoder: map ProtBERT embeddings back to amino acid sequences."""

import torch
import torch.nn as nn

AA_VOCAB: list[str] = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs, alphabetical


class EmbeddingDecoder(nn.Module):
    """Frozen linear projection decoder: 1024-dim embedding → 20-way softmax → sample amino acid."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1024, 20)
        for param in self.parameters():
            param.requires_grad = False

    def decode(self, embedding: torch.Tensor) -> str:
        """Map a single 1024-dim vector to one amino acid character.

        Args:
            embedding: 1-D tensor of shape (1024,)

        Returns:
            Single amino acid character from AA_VOCAB
        """
        with torch.no_grad():
            logits = self.proj(embedding.float())
            probs = torch.softmax(logits, dim=-1)
            idx = int(torch.multinomial(probs, 1).item())
        return AA_VOCAB[idx]

    def decode_sequence(self, embeddings: torch.Tensor) -> str:
        """Map (N, 1024) tensor to amino acid string of length N.

        Args:
            embeddings: 2-D tensor of shape (N, 1024)

        Returns:
            String of N amino acid characters
        """
        return "".join(self.decode(embeddings[i]) for i in range(len(embeddings)))
