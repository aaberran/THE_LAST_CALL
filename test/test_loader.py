import json
from src.loader import validate_functions, validate_prompt, validate_output


VALID_FUNCTIONS_PATH = "data/input/functions_definition.json"
VALID_PROMPTS_PATH = "data/input/function_calling_tests.json"


def test_validate_functions_valid() -> None:
    """Test loading valid functions definition file."""
    result = validate_functions(VALID_FUNCTIONS_PATH)
    assert result is not None
    assert len(result) > 0
    assert result[0].name == "fn_add_numbers"


def test_validate_functions_missing_file() -> None:
    """Test that missing file returns None."""
    result = validate_functions("notfound.json")
    assert result is None


def test_validate_functions_invalid_name(tmp_path) -> None:
    """Test that invalid function name returns None."""
    bad_data = [
        {
            "name": "443,invalid",
            "description": "test",
            "parameters": {},
            "returns": {"type": "number"}
        }
    ]
    bad_file = tmp_path / "bad_name.json"
    bad_file.write_text(json.dumps(bad_data))
    result = validate_functions(str(bad_file))
    assert result is None


def test_validate_prompt_valid() -> None:
    """Test loading valid prompts file."""
    result = validate_prompt(VALID_PROMPTS_PATH)
    assert result is not None
    assert len(result) > 0


def test_validate_prompt_missing_file() -> None:
    """Test that missing prompts file returns None."""
    result = validate_prompt("notfound.json")
    assert result is None


def test_validate_prompt_invalid_json(tmp_path) -> None:
    """Test that invalid JSON prompts returns None."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{bad json}")
    result = validate_prompt(str(bad_file))
    assert result is None


def test_validate_output_valid() -> None:
    """Test creating valid output result."""
    result = validate_output(
        prompt="What is the sum of 3 and 7?",
        name="fn_add_numbers",
        parameters={"a": 3.0, "b": 7.0}
    )
    assert result is not None
    assert result.prompt == "What is the sum of 3 and 7?"
    assert result.name == "fn_add_numbers"
    assert result.parameters["a"] == 3.0
    assert result.parameters["b"] == 7.0


def test_validate_output_string_params() -> None:
    """Test creating output with string parameters."""
    result = validate_output(
        prompt="Greet john",
        name="fn_greet",
        parameters={"name": "john"}
    )
    assert result is not None
    assert result.parameters["name"] == "john"
