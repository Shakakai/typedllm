import os
import pytest

from typedllm import LLMModel


@pytest.fixture(name="openai_key")
def get_openai_key() -> str:
    return os.getenv("OPENAI_API_KEY")


@pytest.fixture(name="model")
def fixture_model(openai_key: str) -> LLMModel:
    return LLMModel(
        name="gpt-4",
        api_key=openai_key,
    )


@pytest.fixture(name="prompt")
def get_prompt_value():
    return {
        "system": "You are an AI assistant. You must respond to questions and follow all instructions.",
        "prompt": "Please respond with just 'Hi' to this message"
    }


@pytest.fixture(name="hi_acompletion")
def hi_acompletion():
    return {"choices": [{"text": "Hi"}]}
