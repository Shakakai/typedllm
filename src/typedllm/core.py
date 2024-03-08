from typing import List, Optional, Dict
from pydantic import BaseModel, Field

from .message import (
    LLMMessage,
    LLMUserMessage,
    LLMToolResultMessage,
    LLMAssistantToolCall,
    LLMAssistantMessage
)
from .tool import Tool


class LLMModel(BaseModel):
    name: str = Field(
        description="The name of the model using the naming conventions of LiteLLM. "
                    "More info here: https://docs.litellm.ai/docs/providers"
    )
    api_key: str = Field(description="The API key for the model provider.")
    organization: Optional[str] = Field(
        description="The organization ID for the model provider.",
        default=None
    )
    api_base: Optional[str] = Field(
        description="If using a Proxy or 3rd Party vendor, this is the base url for requests.",
        default=None
    )
    headers: Optional[Dict[str, str]] = Field(
        description="Additional headers to include with the requests. "
                    "More info here: https://docs.litellm.ai/docs/providers/openai#using-helicone-proxy-with-litellm",
        default=None
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries for a step in the pipeline."
    )
    ssl_verify: bool = Field(
        description="Whether to verify the SSL certificate of the model provider.",
        default=True
    )


class LLMSession(BaseModel):
    model: LLMModel = Field(description="The model used in the session")
    messages: List[LLMMessage] = Field(description="The messages in the session", default=[])
    tools: List[Tool] = Field(description="The tools available in the session", default=[])


class LLMRequest(BaseModel):
    message: Optional[LLMUserMessage] = Field(description="The prompt to use for the request", default=None)
    tool_results: List[LLMToolResultMessage] = Field(description="The tools to use for the request", default=[])
    tools: List[Tool] = Field(
        description="The tools to use for the request in addition to the session tools.",
        default=[]
    )
    required_tool: Optional[Tool] = Field(
        description="The tool that must be used for the request",
        default=None
    )
    force_text_response: bool = Field(
        description="Whether to force the response to be text.",
        default=False
    )


class LLMResponse(BaseModel):
    message: Optional[LLMAssistantMessage] = Field(description="The message from the assistant", default=None)
    tool_calls: List[LLMAssistantToolCall] = Field(description="The tools to use for the request", default=[])
    raw: "ModelResponse" = Field(description="The raw response from the model provider")

