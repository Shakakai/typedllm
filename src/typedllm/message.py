import json
from typing import List, Union, Any
from pydantic import BaseModel, Field
from .tool import Tool


class LLMMessage(BaseModel):

    def openai_json(self):
        raise NotImplementedError("This method must be implemented by the subclass.")


class LLMSystemMessage(LLMMessage):
    content: str

    def openai_json(self):
        return {
            "role": "system",
            "content": self.content
        }


class ImageURL(BaseModel):
    url: str = Field(description="The URL of the image. Can be a web URL or a Base64 encoded image.")


class LLMUserMessage(LLMMessage):
    content: Union[str, List[Union[str, ImageURL]]] = Field(
        description="The content of the message. Either a list of strings or Image URLs."
    )

    def openai_json(self):
        content = None
        if isinstance(self.content, str):
            content = self.content
        else:
            content = []
            for item in self.content:
                if isinstance(item, str):
                    content.append({
                        "type": "text",
                        "text": item
                    })
                else:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": item.url
                        }
                    })
        return {
            "role": "user",
            "content": content
        }


class LLMAssistantToolCall(LLMMessage):
    id: str = Field(description="The id of the tool call")
    tool: Tool
    args: str = Field(description="The arguments to the function")

    @property
    def arguments(self):
        argument_dict = json.loads(self.args)
        clz = self.tool.parameter_type
        argument = clz(**argument_dict)
        return argument

    def openai_json(self):
        return {
            "id": self.id,
            "function": {
                "name": self.tool.name,
                "arguments": self.args
            }
        }


class LLMAssistantMessage(LLMMessage):
    content: str = Field(
        description="The content of the message. Either a list of strings or Image URLs."
    )

    def openai_json(self):
        return {
            "role": "assistant",
            "content": self.content
        }


class LLMToolResultMessage(LLMMessage):
    id: str = Field(description="The id of the tool call")
    content: Any = Field(description="The result of the tool call")

    def openai_json(self):
        return {
            "role": "tool",
            "content": self.content,
            "tool_call_id": self.id
        }
