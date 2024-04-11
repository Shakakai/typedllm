from pydantic import BaseModel
from typedllm.tool import Tool, create_tool_from_function


class TestTool(Tool):
    value: int


def test_basic_tool_creation():
    result = TestTool.openapi_json()

    assert result == {
        "type": "function",
        "function": {
            "name": "TestTool",
            "description": "Tool called TestTool",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "title": "Value",
                        "type": "integer"
                    }
                },
                "required": ["value"]
            }
        }
    }


def test_tool_creation_from_function():
    def test_function(value: int) -> str:
        return str(value)

    FuncTool = create_tool_from_function(test_function)

    result = FuncTool.openapi_json()

    assert FuncTool.__name__ == "test_function"
    assert FuncTool.__doc__ == "Tool called test_function"

    assert result == {
        "type": "function",
        "function": {
            "name": "test_function",
            "description": "Tool called test_function",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "title": "Value",
                        "type": "integer"
                    }
                },
                "required": ["value"]
            }
        }
    }
