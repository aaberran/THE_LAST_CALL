from src.schema import ResultValidation
from src.constraints import get_valid_next_tokens, mask_logits
from typing import Any
import numpy as np
import re


def build_prompt(user_prompt: str, functions: list) -> str:
    """Build a prompt string for the LLM.

    Args:
        user_prompt: Natural language request from user.
        functions: List of available function definitions.

    Returns:
        Formatted prompt string.
    """
    result = "You are a function calling assistant.\n"
    result += "Available functions:\n"

    for func in functions:
        name = func.name
        desc = func.description
        params = []
        for k, v in func.parameters.items():
            params.append(f"{k} ({v.type})")
        result += f"{name} - {desc} - Parameters: {", ".join(params)}\n"

    result += f"\nUser request: {user_prompt}\n"
    result += '\nResponse:\n'

    return result


def generate_stops_and_numbers(
    str_to_id: dict[str, int]
) -> dict[str, set[int]]:
    """Precompute stop tokens and number tokens from vocabulary.

    Args:
        str_to_id: Vocabulary mapping string to token id.

    Returns:
        Dictionary with stop_token and all_tokens sets.
    """
    stop_token = set()
    stop = [",", "}", "Ġ", " ", " ,", "\n"]
    stop_token = {str_to_id[c] for c in stop if c in str_to_id}
    numbers_tokens = set()
    for k, v in str_to_id.items():
        if (all(c in "0123456789.-" for c in k)
                and k.count('.') <= 1 and k.count('-') <= 1):
            numbers_tokens.add(v)
    result = {
        "stop_token": stop_token,
        "all_tokens": numbers_tokens
    }
    return result


def generate_number_value(
    input_ids: list[int],
    model: Any,
    str_to_id: dict[str, int],
    id_to_str: dict[int, str]
) -> tuple[float, list[int]]:
    """Generate a numeric value using constrained decoding.

    Args:
        input_ids: Current token sequence.
        model: The LLM model instance.
        str_to_id: Vocabulary mapping string to token id.
        id_to_str: Vocabulary mapping token id to string.

    Returns:
        Tuple of (float value, generated token ids).
    """
    all_valid = generate_stops_and_numbers(str_to_id)
    curr_number = ""
    gen_ids: list[int] = []
    while True:
        full_ids = input_ids + gen_ids
        logits = model.get_logits_from_input_ids(full_ids)
        masked = mask_logits(
            logits, all_valid["all_tokens"] | all_valid["stop_token"])
        max_token = int(np.argmax(masked))
        token_str = id_to_str[max_token]
        if max_token in all_valid["stop_token"]:
            break
        gen_ids.append(max_token)
        curr_number += token_str
    curr_number = curr_number.strip()
    try:
        val = float(curr_number) if curr_number else 0.0
    except ValueError:
        val = 0.0

    return val, gen_ids,


def generate_string_value(
    input_ids: list[int],
    model: Any,
    id_to_str: dict[int, str]
) -> tuple[str, list[int]]:
    """Generate a string value using constrained decoding.

    Args:
        input_ids: Current token sequence.
        model: The LLM model instance.
        id_to_str: Vocabulary mapping token id to string.

    Returns:
        Tuple of (string value, generated token ids).
    """
    gen_id: list[int] = []
    result: str = ""
    while True:
        full_id = input_ids + gen_id
        logits = model.get_logits_from_input_ids(full_id)
        max_token: int = int(np.argmax(logits))
        if '"' in id_to_str[int(max_token)]:
            break
        gen_id.append(max_token)
        result += id_to_str[max_token]

    return result.replace("Ġ", " ").replace("\\'", "'"), gen_id


def is_valid_prompt(prompt: str) -> bool:
    """Check if a prompt is valid and meaningful.

    Args:
        prompt: User input string to validate.

    Returns:
        True if prompt appears valid, False otherwise.
    """
    if len(prompt.strip()) < 3:
        return False

    words = prompt.strip().split()
    real_words = [
        w for w in words if w.isalpha() and len(w) > 12 and not w.isdigit()]
    if len(real_words) > 0:
        return False
    return True


def generate_function_call(
    prompt: Any,
    functions: list,
    model: Any,
    str_to_id: dict[str, int],
    id_to_str: dict[int, str]
) -> ResultValidation:
    """Generate a function call from a natural language prompt.

    Args:
        prompt: PromptValidate object containing the user request.
        functions: List of available function definitions.
        model: The LLM model instance.
        str_to_id: Vocabulary mapping string to token id.
        id_to_str: Vocabulary mapping token id to string.

    Returns:
        ResultValidation object with function name and parameters.
    """
    if not is_valid_prompt(prompt.prompt):
        return ResultValidation(
            prompt=prompt.prompt, name="Invalid!",
            parameters={"error": "prompt not recognized"})
    functions_names = [fn.name for fn in functions]
    curr_str = ""
    gen_ids: list[int] = []
    prompt_text = build_prompt(prompt, functions)
    input_ids = model.encode(prompt_text)[0].tolist()
    input_ids.append(90)
    input_ids.append(1)
    input_ids.append(606)
    input_ids.append(788)
    input_ids.append(220)
    input_ids.append(1)
    for _ in range(30):
        full_input_ids = input_ids + gen_ids

        logits = model.get_logits_from_input_ids(full_input_ids)
        logits_numpy = np.array(logits)

        valid_token_ids = get_valid_next_tokens(
            curr_str, functions_names, str_to_id)
        masked_logits = mask_logits(logits_numpy, valid_token_ids)
        next_token_id = int(np.argmax(masked_logits))
        token_str = id_to_str[next_token_id]
        gen_ids.append(next_token_id)

        curr_str += token_str
        if curr_str in functions_names:
            chosen_function = next(
                fn for fn in functions if fn.name == curr_str)
            input_ids.extend(gen_ids)
            input_ids.append(str_to_id['"'])
            input_ids.append(str_to_id[','])
            input_ids.append(str_to_id['Ġ'])
            input_ids.append(str_to_id['"'])
            input_ids.append(13786)
            input_ids.append(str_to_id['":'])
            input_ids.append(str_to_id['Ġ'])
            input_ids.append(str_to_id['{'])
            params = list(chosen_function.parameters.items())
            parameters: dict[str, float | str | int] = {}
            for i, (name, value) in enumerate(params):
                input_ids.append(str_to_id['"'])
                for chr in model.encode(name)[0].tolist():
                    input_ids.append(chr)
                input_ids.append(str_to_id['"'])
                input_ids.append(str_to_id[':'])
                input_ids.append(str_to_id['Ġ'])
                if value.type == "number":
                    has_negative = bool(re.search(r"-\d+", prompt.prompt))
                    if has_negative:
                        numb = re.findall(
                            r"-?\d+(?:\.\d+)?", prompt.prompt)
                        parameters[name] = float(numb[i])
                    else:
                        val, ids = generate_number_value(
                            input_ids, model, str_to_id, id_to_str)
                        parameters[name] = val
                        input_ids += ids
                    if i < len(params) - 1:
                        input_ids.append(str_to_id[','])
                        input_ids.append(str_to_id['Ġ'])
                    else:
                        input_ids.append(str_to_id['}'])
                        input_ids.append(str_to_id['}'])
                elif value.type == "integer":
                    has_negative = bool(re.search(r"-\d+", prompt.prompt))
                    if has_negative:
                        numb = re.findall(
                            r"-?\d+(?:\.\d+)?", prompt.prompt)
                        parameters[name] = int(numb[i])
                    else:
                        val, ids = generate_number_value(
                            input_ids, model, str_to_id, id_to_str)
                        parameters[name] = int(val)
                        input_ids += ids
                    if i < len(params) - 1:
                        input_ids.append(str_to_id[','])
                        input_ids.append(str_to_id['Ġ'])
                    else:
                        input_ids.append(str_to_id['}'])
                        input_ids.append(str_to_id['}'])
                elif value.type == "string":
                    input_ids.append(str_to_id['"'])
                    (str_val, ids) = generate_string_value(
                        input_ids, model, id_to_str)
                    parameters[name] = str_val
                    input_ids += ids
            return ResultValidation(
                prompt=prompt.prompt,
                name=curr_str,
                parameters=parameters
            )
    return ResultValidation(
        prompt=prompt.prompt,
        name="invalid",
        parameters={"error": "no matching function found"}
    )
