import pytest
from pydantic import ValidationError
from src.schema import (
    PromptValidate,
    TypeClass,
    ValidateSchema,
    ResultValidation
)


def test_prompt_validate_valid() -> None:
    """Test valid prompt creation."""
    p = PromptValidate(prompt="Greet john")
    assert p.prompt == "Greet john"


def test_prompt_validate_empty() -> None:
    """Test that empty prompt raises ValidationError."""
    with pytest.raises(ValidationError):
        PromptValidate(prompt="")


def test_type_class_valid() -> None:
    """Test valid TypeClass creation."""
    t = TypeClass(type="number")
    assert t.type == "number"


def test_validate_schema_valid() -> None:
    """Test valid function schema creation."""
    schema = ValidateSchema(
        name="fn_add_numbers",
        description="Add two numbers",
        parameters={
            "a": TypeClass(type="number"),
            "b": TypeClass(type="number")
        },
        returns=TypeClass(type="number")
    )
    assert schema.name == "fn_add_numbers"
    assert "a" in schema.parameters


def test_validate_schema_invalid_name() -> None:
    """Test that invalid function name raises ValidationError."""
    with pytest.raises(ValidationError):
        ValidateSchema(
            name="443,invalid",
            description="test",
            parameters={},
            returns=TypeClass(type="number")
        )


def test_validate_schema_valid_name_formats() -> None:
    """Test various valid function name formats."""
    valid_names = ["fn_add", "myFunction", "func123", "a_b_c"]
    for name in valid_names:
        schema = ValidateSchema(
            name=name,
            description="test",
            parameters={},
            returns=TypeClass(type="number")
        )
        assert schema.name == name


def test_result_validation_valid() -> None:
    """Test valid result creation."""
    r = ResultValidation(
        prompt="test",
        name="fn_add_numbers",
        parameters={"a": 3.0, "b": 7.0}
    )
    assert r.name == "fn_add_numbers"
    assert r.parameters["a"] == 3.0


def test_result_validation_mixed_params() -> None:
    """Test result with mixed parameter types."""
    r = ResultValidation(
        prompt="test",
        name="fn_greet",
        parameters={"name": "john"}
    )
    assert r.parameters["name"] == "john"
