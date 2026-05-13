import json
import numpy as np
from numpy import ndarray


def load_vocabulary(vocab_path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary from JSON file.
    Args:
        vocab_path: Path to vocabulary JSON file.
    Returns:
        Tuple of (token_string to id, id to token_string) dictionaries.
    """
    try:
        with open(vocab_path, 'r') as f:
            token_id = json.load(f)
        id_token = {v: k for k, v in token_id.items()}
        return token_id, id_token
    except Exception as e:
        print(e)
        raise


def functions_names(path: str) -> list[str]:
    """Extract function names from a functions definition file.
    Args:
        path: Path to functions definition JSON file.
    Returns:
        List of function name strings.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    names_list = [item["name"] for item in data]
    return names_list


def mask_logits(
    logits: list[float] | ndarray,
    valid_token_ids: set[int]
) -> ndarray:

    """Mask logits to only allow valid token ids.

    Args:
        logits: Raw logits from the model.
        valid_token_ids: Set of token ids to keep.

    Returns:
        Masked logits array with -inf for invalid tokens.
    """
    logits = np.array(logits)
    logits_shadow = np.full_like(logits, -np.inf)
    valid_ids = list(valid_token_ids)
    logits_shadow[valid_ids] = logits[valid_ids]
    return logits_shadow


def get_token_id_for_string(s: str, str_to_id: dict[str, int]) -> int | None:
    """Look up token id for an exact string.

    Args:
        s: Token string to look up.
        str_to_id: Vocabulary mapping string to token id.

    Returns:
        Token id if found, None otherwise.
    """
    return str_to_id.get(s, None)


def get_valid_next_tokens(
    current_prefix: str, valid_strings: list[str],
    str_to_id: dict[str, int]
) -> set[int]:

    """Get valid next token ids based on current prefix.
    Args:
        current_prefix: String generated so far.
        valid_strings: List of valid complete strings.
        str_to_id: Vocabulary mapping string to token id.
        Returns:
        Set of valid token ids for the next generation step.
    """

    items = [i for i in valid_strings if i.startswith(current_prefix)]
    valid_tokens = set()
    for s, i in str_to_id.items():
        possible = current_prefix + s
        for it in items:
            if it.startswith(possible):
                valid_tokens.add(i)
                break
        if possible in items:
            break
    return valid_tokens
