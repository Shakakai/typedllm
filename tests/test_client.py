from typing import Type

import pytest

from pydantic import BaseModel, Field

from typedllm import (
    LLMModel,
    LLMSession,
    LLMRequest,
    LLMUserMessage,
    llm_request,
    async_llm_request,
    create_tool_from_function,
    Tool
)


@pytest.mark.asyncio
async def test_basic_request(openai_key: str):
    model = LLMModel(
        name="gpt-4-0125-preview",
        api_key=openai_key,
    )
    session = LLMSession(
        model=model,
    )
    request = LLMRequest(
        message=LLMUserMessage(
            content="What year was New York City founded?",
        ),
        force_text_response=True
    )
    session, response = await async_llm_request(session, request)
    assert response.message.content.index("1624") > -1


class FoundingYear(BaseModel):
    """This represents a city and includes the year the city was founded."""
    year: int = Field(description="The year the city was founded.")
    city: str = Field(description="The name of the city.")


def get_city_founding_history(city: FoundingYear) -> str:
    return (f"Historical weather for {city.city} since its founding in {city.year}. "
            f"The weather is very nice. Average 80 degrees F.")


def test_make_tool_from_function():
    Tool = create_tool_from_function(get_city_founding_history)
    assert Tool.__name__ == "get_city_founding_history"
    assert "city" in Tool.model_fields
    assert Tool.model_fields["city"].annotation == FoundingYear


@pytest.mark.asyncio
async def test_basic_tools_request(openai_key: str):
    model = LLMModel(
        name="gpt-4",
        api_key=openai_key,
    )
    tool: Type[Tool] = create_tool_from_function(get_city_founding_history)
    session = LLMSession(
        model=model,
        tools=[tool]
    )
    request = LLMRequest(
        message=LLMUserMessage(
            content="What is the founding story of New York City?",
        ),
        required_tool=tool,
    )
    session, response = await async_llm_request(session, request)
    assert session
    assert response
    assert response.tool_calls[0].tool.__class__.__name__ == "get_city_founding_history"
    assert response.tool_calls[0].tool.city.city.index("New York") > -1
    assert response.tool_calls[0].tool.city.year == 1624


@pytest.mark.asyncio
async def test_basic_lvm_request(openai_key: str):
    pass
