from litellm.utils import ModelResponse
from .interop import litellm_request
from .message import LLMAssistantMessage, LLMAssistantToolCall
from .tool import ToolCollection
from .core import LLMSession, LLMRequest, LLMResponse


async def llm_request(
        session: LLMSession,
        request: LLMRequest,
        verbose: bool = False
) -> (LLMSession, LLMResponse):
    # TODO: Add support for a Memory Interface here.
    # Memory can be set as a configuration option and can be invoked here.
    messages = [
        *session.messages,
        *request.tool_results,
        request.message
    ]
    messages = [msg.openai_json() for msg in messages]
    tools = ToolCollection(*session.tools, *request.tools)
    tools_json = tools.openapi_json()
    if len(tools) == 0 or request.force_text_response:
        tool_choice = "none"
    elif request.required_tool:
        tool_choice = request.required_tool.openai_tool_choice_json()
    else:
        tool_choice = "auto"

    raw_response = await litellm_request(
        session.model,
        messages,
        tools_json,
        tool_choice,
        verbose=verbose
    )
    response = extract_response_messages(raw_response, tools)

    # Add tool calls to session messsages
    if response.tool_calls and len(response.tool_calls) > 0:
        for tool_call in response.tool_calls:
            session.messages.append(tool_call)


    # Add messages to the session
    if response.message:
        session.messages.append(response.message)

    return session, response


def extract_response_messages(response: ModelResponse, tools: ToolCollection) -> LLMResponse:
    if len(response.choices) != 1:
        raise Exception("Invalid number of choices in response. Expect only one choice.")

    msg = response.choices[0].message

    if msg["role"] != "assistant":
        raise Exception("Invalid role in response")
    llm_msg = None
    if msg.content:
        llm_msg = LLMAssistantMessage(
            content=msg.content,
        )
    if not msg.tool_calls:
        return LLMResponse(
            message=llm_msg,
            raw=response
        )
    tool_calls = []
    if len(msg.tool_calls) > 0:
        for tool_call in msg["tool_calls"]:
            tc = LLMAssistantToolCall(
                id=tool_call.id,
                tool=tools.get_by_name(tool_call.function.name),
                args=tool_call.function.arguments
            )
            tool_calls.append(tc)

    return LLMResponse(
        message=llm_msg,
        tool_calls=tool_calls,
        raw=response
    )
