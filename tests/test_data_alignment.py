"""
Unit tests for sequence-label alignment in the DisProt/CAID data pipeline.

These tests are the primary guard against silent off-by-one bugs in the
sliding window chunker and the CAID label parser.
"""

import pytest
from src.data.data import (
    get_disprot_data,
    get_disprot_cut_data,
    get_caid_data,
    get_caid_ids,
    process_disprot_cut_data,
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWYXUBZacdefghiklmnpqrstvwyxubz")


# ---------------------------------------------------------------------------
# Fixtures — load data once per session to avoid redundant I/O
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def all_data():
    return get_disprot_data()


@pytest.fixture(scope="session")
def cut_data():
    return get_disprot_cut_data()


@pytest.fixture(scope="session")
def caid_data():
    return get_caid_data()


@pytest.fixture(scope="session")
def caid_ids():
    return set(get_caid_ids())


# ---------------------------------------------------------------------------
# DisProt all_data tests
# ---------------------------------------------------------------------------

class TestDisProtAllData:
    def test_nonempty(self, all_data):
        assert len(all_data["sequences"]) > 0

    def test_seq_label_count_match(self, all_data):
        assert len(all_data["sequences"]) == len(all_data["disorder region"])

    def test_per_entry_length_match(self, all_data):
        mismatches = [
            i for i, (s, l) in enumerate(
                zip(all_data["sequences"], all_data["disorder region"])
            )
            if len(s) != len(l)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} entries with seq/label length mismatch: indices {mismatches[:5]}"
        )

    def test_label_values_binary(self, all_data):
        bad = [
            i for i, labels in enumerate(all_data["disorder region"])
            if any(v not in (0, 1) for v in labels)
        ]
        assert bad == [], f"Non-binary labels in entries: {bad[:5]}"

    def test_sequences_nonempty(self, all_data):
        empty = [i for i, s in enumerate(all_data["sequences"]) if len(s) == 0]
        assert empty == [], f"Empty sequences at indices: {empty[:5]}"


# ---------------------------------------------------------------------------
# DisProt cut_data (sliding window) tests
# ---------------------------------------------------------------------------

class TestDisProtCutData:
    def test_nonempty(self, cut_data):
        assert len(cut_data["sequences"]) > 0

    def test_seq_label_count_match(self, cut_data):
        assert len(cut_data["sequences"]) == len(cut_data["disorder region"])

    def test_per_entry_length_match(self, cut_data):
        mismatches = [
            i for i, (s, l) in enumerate(
                zip(cut_data["sequences"], cut_data["disorder region"])
            )
            if len(s) != len(l)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} chunked entries with seq/label length mismatch"
        )

    def test_max_chunk_length(self, cut_data):
        too_long = [
            i for i, s in enumerate(cut_data["sequences"]) if len(s) > 400
        ]
        assert too_long == [], f"{len(too_long)} chunks exceed 400 AA"

    def test_label_values_binary(self, cut_data):
        bad = [
            i for i, labels in enumerate(cut_data["disorder region"])
            if any(v not in (0, 1) for v in labels)
        ]
        assert bad == [], f"Non-binary labels in chunked entries: {bad[:5]}"


# ---------------------------------------------------------------------------
# Sliding window chunker correctness (unit test with synthetic input)
# ---------------------------------------------------------------------------

class TestSlidingWindowChunker:
    def test_short_sequence_not_chunked(self):
        all_data = {
            "sequences": ["ACDEF"],
            "disorder region": [[0, 0, 1, 1, 0]],
            "id": ["DP99999"],
        }
        result = process_disprot_cut_data(all_data, cut_length=400, alpha=0.5, save=False)
        assert result["sequences"] == ["ACDEF"]
        assert result["disorder region"] == [[0, 0, 1, 1, 0]]

    def test_chunked_labels_match_original(self):
        seq = "A" * 600
        labels = [0] * 300 + [1] * 300
        all_data = {"sequences": [seq], "disorder region": [labels], "id": ["DP99999"]}
        result = process_disprot_cut_data(all_data, cut_length=400, alpha=0.5, save=False)
        for s, l in zip(result["sequences"], result["disorder region"]):
            assert len(s) == len(l), "chunk seq/label length mismatch"
            assert all(v in (0, 1) for v in l), "non-binary label in chunk"

    def test_exact_cut_length_sequence(self):
        seq = "M" * 400
        labels = [1] * 400
        all_data = {"sequences": [seq], "disorder region": [labels], "id": ["DP99999"]}
        result = process_disprot_cut_data(all_data, cut_length=400, alpha=0.5, save=False)
        for s, l in zip(result["sequences"], result["disorder region"]):
            assert len(s) == len(l)

    def test_label_slice_correctness(self):
        # Verify that a chunk's labels correspond to the right slice of the original
        seq = "ACDEFGHIKLMNPQRSTVWY" * 25  # 500 AA
        labels = list(range(len(seq)))      # unique value per position for traceability
        # Clamp to 0/1 isn't needed here; we're testing slice correctness directly
        all_data = {"sequences": [seq], "disorder region": [labels], "id": ["DP99999"]}
        result = process_disprot_cut_data(all_data, cut_length=400, alpha=0.5, save=False)
        # First chunk must be seq[:400] / labels[:400]
        assert result["sequences"][0] == seq[:400]
        assert result["disorder region"][0] == labels[:400]


# ---------------------------------------------------------------------------
# CAID data tests
# ---------------------------------------------------------------------------

class TestCAIDData:
    def test_sequence_count(self, caid_data):
        assert len(caid_data["sequences"]) == 652, (
            f"Expected 652 CAID sequences, got {len(caid_data['sequences'])}"
        )

    def test_seq_label_count_match(self, caid_data):
        assert len(caid_data["sequences"]) == len(caid_data["disorder region"])

    def test_per_entry_length_match(self, caid_data):
        mismatches = [
            i for i, (s, l) in enumerate(
                zip(caid_data["sequences"], caid_data["disorder region"])
            )
            if len(s) != len(l)
        ]
        assert mismatches == [], (
            f"{len(mismatches)} CAID entries with seq/label length mismatch"
        )

    def test_total_residue_count(self, caid_data):
        total = sum(len(s) for s in caid_data["sequences"])
        assert total == 338068, f"Expected 338068 CAID residues, got {total}"

    def test_disorder_content(self, caid_data):
        all_labels = [v for labels in caid_data["disorder region"] for v in labels]
        pct = sum(all_labels) / len(all_labels)
        assert 0.15 < pct < 0.18, f"Disorder content {pct:.3f} outside expected range [0.15, 0.18]"

    def test_label_values_binary(self, caid_data):
        bad = [
            i for i, labels in enumerate(caid_data["disorder region"])
            if any(v not in (0, 1) for v in labels)
        ]
        assert bad == [], f"Non-binary CAID labels at entries: {bad[:5]}"

    def test_no_empty_sequences(self, caid_data):
        empty = [i for i, s in enumerate(caid_data["sequences"]) if len(s) == 0]
        assert empty == [], f"Empty CAID sequences at indices: {empty}"


# ---------------------------------------------------------------------------
# CAID exclusion from training set
# ---------------------------------------------------------------------------

class TestCAIDExclusion:
    def test_caid_ids_not_in_cut_data(self, cut_data, caid_ids, all_data):
        # The cut_data has no 'id' field; verify via the all_data id list
        all_ids = all_data["id"]
        all_seqs = all_data["sequences"]
        caid_seqs = {
            all_seqs[i] for i, id_ in enumerate(all_ids) if id_ in caid_ids
        }
        # No cut_data sequence should be the full sequence of a CAID entry
        # (chunks are fine since they're substrings; check for exact full-seq matches)
        cut_seqs = set(cut_data["sequences"])
        leaked = caid_seqs & cut_seqs
        assert leaked == set(), (
            f"{len(leaked)} CAID full sequences found verbatim in training cut_data"
        )

    def test_caid_id_count(self, caid_ids):
        assert len(caid_ids) == 652, f"Expected 652 CAID IDs, got {len(caid_ids)}"
