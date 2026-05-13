import pytest
from src.generator import build_prompt, is_valid_prompt
from src.schema import ValidateSchema, TypeClass


@pytest.fixture
def sample_functions() -> list:
    """Create sample functions for testing."""
    return [
        ValidateSchema(
            name="fn_add_numbers",
            description="Add two numbers",
            parameters={
                "a": TypeClass(type="number"),
                "b": TypeClass(type="number")
            },
            returns=TypeClass(type="number")
        ),
        ValidateSchema(
            name="fn_greet",
            description="Greet a person",
            parameters={
                "name": TypeClass(type="string")
            },
            returns=TypeClass(type="string")
        )
    ]


def test_build_prompt_contains_function_names(sample_functions) -> None:
    """Test that prompt contains all function names."""
    prompt = build_prompt("test request", sample_functions)
    assert "fn_add_numbers" in prompt
    assert "fn_greet" in prompt


def test_build_prompt_contains_user_request(sample_functions) -> None:
    """Test that prompt contains the user request."""
    prompt = build_prompt("What is the sum of 3 and 7?", sample_functions)
    assert "What is the sum of 3 and 7?" in prompt


def test_build_prompt_ends_with_json_prefix(sample_functions) -> None:
    """Test that prompt ends with Response section."""
    prompt = build_prompt("test", sample_functions)
    assert "Response:" in prompt
    assert "User request: test" in prompt


def test_build_prompt_contains_parameters(sample_functions) -> None:
    """Test that prompt contains parameter descriptions."""
    prompt = build_prompt("test", sample_functions)
    assert "number" in prompt
    assert "string" in prompt


def test_is_valid_prompt_valid() -> None:
    """Test valid prompts pass validation."""
    assert is_valid_prompt("Greet john") is True
    assert is_valid_prompt("What is the sum of 3 and 7?") is True
    assert is_valid_prompt("Reverse the string hello") is True


def test_is_valid_prompt_too_short() -> None:
    """Test that short prompts fail validation."""
    assert is_valid_prompt("") is False
    assert is_valid_prompt("ab") is False
    assert is_valid_prompt("  ") is False


def test_is_valid_prompt_invalid() -> None:
    """Test that random character strings fail validation."""
    assert is_valid_prompt("kibfiuberfrfrg") is False
    assert is_valid_prompt("frvrvwrtgevcwvlinfvfr") is False
