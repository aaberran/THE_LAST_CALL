from src.schema import ValidateSchema, PromptValidate, ResultValidation
import json
from pydantic import ValidationError
from typing import Optional


def validate_functions(
    functions_definition_path: str
) -> Optional[list[ValidateSchema]]:
    """Load and validate function definitions from a JSON file.

    Args:
        functions_definition_path: Path to the functions definition JSON file.

    Returns:
        List of ValidateSchema objects or None if validation fails.
    """
    try:
        with open(functions_definition_path) as f:
            json_data = json.load(f)
        data = [ValidateSchema(**item) for item in json_data]
        return data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        return None
    except ValidationError as e:
        print(f"Error: {e.errors()[0]["msg"]}")
        return None


def validate_prompt(
    prompt_path: str
) -> Optional[list[PromptValidate]]:
    """Load and validate prompts from a JSON file.

    Args:
        prompt_path: Path to the prompts JSON file.

    Returns:
        List of PromptValidate objects or None if validation fails.
    """
    try:
        with open(prompt_path) as f:
            json_data = json.load(f)
        data = [PromptValidate(**item) for item in json_data]
        return data
    except (FileNotFoundError, ) as e:
        print(f"Error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: {e}")
        return None
    except ValidationError as e:
        print(f"Error: {e.errors()[0]["msg"]}")
        return None


def validate_output(
    prompt: str,
    name: str,
    parameters: dict[str, float | str | int]
) -> Optional[ResultValidation]:
    """Validate and create a ResultValidation object.

    Args:
        prompt: Original user prompt.
        name: Function name to call.
        parameters: Dictionary of parameter names and values.

    Returns:
        ResultValidation object or None if validation fails.
    """
    try:
        result: ResultValidation = ResultValidation(
            prompt=prompt,
            name=name,
            parameters=parameters
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None
    except ValidationError as e:
        print(f"Error: {e.errors()[0]["msg"]}")
        return None
