from typing import List, Any, Dict
import httpx
import litellm
from litellm import acompletion
from litellm.utils import ModelResponse


async def litellm_request(
        model: "LLMModel",
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        tool_choice: Dict[str, Any],
        verbose: bool = False
) -> ModelResponse:

    if len(messages) == 0:
        raise ValueError("No messages provided")

    if not model.ssl_verify:
        litellm.client_session = httpx.Client(verify=False)

    if verbose:
        litellm.set_verbose = True

    if model.headers:
        litellm.headers = model.headers

    req = {
        "model": model.name,
        "messages": messages,
        "max_retries": model.max_retries,
        "api_key": model.api_key,
    }

    if len(tools) > 0:
        req["tools"] = tools
        req["tool_choice"] = tool_choice

    if model.organization:
        req["organization"] = model.organization
    if model.api_base:
        req["api_base"] = model.api_base

    response = await acompletion(**req)

    if not model.ssl_verify:
        litellm.client_session = None  # reset to default. Might be unnecessary. Evaluate later.

    if model.headers:
        litellm.headers = None  # reset to default. Might be unnecessary. Evaluate later.

    if verbose:
        litellm.set_verbose = False

    return response
