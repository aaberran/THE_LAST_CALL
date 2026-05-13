from pydantic import BaseModel, field_validator, Field
import keyword
from typing import Literal


class PromptValidate(BaseModel):
    """Model for validating user prompts.

    Attributes:
        prompt: The natural language request string.
    """
    prompt: str = Field(..., min_length=1)


class TypeClass(BaseModel):
    """Model for representing a parameter type.

    Attributes:
        type: The type string (number, string, boolean, etc).
    """
    type: Literal["string", "integer", "boolean", "number"]


class ValidateSchema(BaseModel):
    """Model for validating function definitions.

    Attributes:
        name: Function name.
        description: Function description.
        parameters: Dictionary of parameter names to their types.
        returns: Return type of the function.
    """
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: dict[str, TypeClass]
    returns: TypeClass

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate function name format.

        Args:
            v: The function name to validate.

        Returns:
            The validated function name.

        Raises:
            ValueError: If name contains invalid characters.
        """
        if not v.isidentifier() or keyword.iskeyword(v):
            raise ValueError("Invalid function name format")
        return v


class ResultValidation(BaseModel):
    """Model for validating function call results.

    Attributes:
        prompt: Original user prompt.
        name: Name of the function to call.
        parameters: Dictionary of parameter names and their values.
    """
    prompt: str
    name: str
    parameters: dict[str, bool | str | float | int]
