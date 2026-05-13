import numpy as np
import pytest
from llm_sdk import Small_LLM_Model  # type: ignore
from src.constraints import (
    load_vocabulary,
    mask_logits,
    get_valid_next_tokens,
    get_token_id_for_string
)


@pytest.fixture(scope="module")
def vocab():
    """Load vocabulary once for all tests."""
    model = Small_LLM_Model()
    path = model.get_path_to_vocab_file()
    str_to_id, id_to_str = load_vocabulary(path)
    return str_to_id, id_to_str


def test_load_vocabulary(vocab) -> None:
    """Test that vocabulary loads correctly with both directions."""
    str_to_id, id_to_str = vocab
    assert isinstance(str_to_id, dict)
    assert isinstance(id_to_str, dict)
    assert '"' in str_to_id
    assert str_to_id['"'] == 1
    assert id_to_str[1] == '"'
    assert len(str_to_id) == len(id_to_str)


def test_load_vocabulary_invalid_path() -> None:
    """Test that load_vocabulary handles missing file gracefully."""
    with pytest.raises(Exception):
        load_vocabulary("file/not_found.json")


def test_mask_logits_basic() -> None:
    """Test that mask_logits sets invalid tokens to -inf."""
    logits = [1.0, 2.0, 3.0, 4.0, 5.0]
    valid = {1, 3}
    masked = mask_logits(logits, valid)

    assert masked[0] == -np.inf
    assert masked[1] == 2.0
    assert masked[2] == -np.inf
    assert masked[3] == 4.0
    assert masked[4] == -np.inf


def test_mask_logits_all_valid() -> None:
    """Test mask_logits when all tokens are valid."""
    logits = [1.0, 2.0, 3.0]
    valid = {0, 1, 2}
    masked = mask_logits(logits, valid)

    assert masked[0] == 1.0
    assert masked[1] == 2.0
    assert masked[2] == 3.0


def test_mask_logits_none_valid() -> None:
    """Test mask_logits when no tokens are valid."""
    logits = [1.0, 2.0, 3.0]
    valid: set[int] = set()
    masked = mask_logits(logits, valid)

    assert all(v == -np.inf for v in masked)


def test_get_valid_next_tokens_empty_prefix(vocab) -> None:
    """Test prefix matching with empty prefix returns tokens."""
    str_to_id, _ = vocab
    valid_strings = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]
    result = get_valid_next_tokens("", valid_strings, str_to_id)

    assert isinstance(result, set)
    assert len(result) > 0


def test_get_valid_next_tokens_no_match(vocab) -> None:
    """Test prefix matching returns empty set when no match."""
    str_to_id, _ = vocab
    valid_strings = ["fn_add_numbers", "fn_greet"]
    result = get_valid_next_tokens("xyz_", valid_strings, str_to_id)

    assert len(result) == 0


def test_get_token_id_for_string(vocab) -> None:
    """Test token id lookup for known strings."""
    str_to_id, _ = vocab

    assert get_token_id_for_string('"', str_to_id) == 1
    assert get_token_id_for_string('{', str_to_id) == 90
    assert get_token_id_for_string('}', str_to_id) == 92


def test_get_token_id_for_string_missing(vocab) -> None:
    """Test token id lookup returns None for unknown strings."""
    str_to_id, _ = vocab
    result = get_token_id_for_string("NOTATOKEN123", str_to_id)
    assert result is None
