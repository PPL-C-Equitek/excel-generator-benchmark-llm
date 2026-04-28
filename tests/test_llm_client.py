import httpx
import openai
import pytest

from src.llm_client import LLMAuthError, LLMClient


def _openai_api_error(error_cls, status_code):
    request = httpx.Request(
        "POST",
        "https://sumopod.test/v1/chat/completions",
    )
    response = httpx.Response(status_code=status_code, request=request)
    return error_cls(
        f"OpenAI API returned HTTP {status_code}",
        response=response,
        body={"error": {"message": f"HTTP {status_code}"}},
    )


def _chat_completion_response(mocker, content):
    message = mocker.Mock(content=content)
    choice = mocker.Mock(message=message)
    return mocker.Mock(choices=[choice])


@pytest.fixture
def openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://sumopod.test/v1")


def test_llm_client_initializes_openai_and_returns_generated_text(
    mocker,
    openai_env,
):
    openai_client = mocker.Mock()
    openai_client.chat.completions.create.return_value = _chat_completion_response(
        mocker,
        "The benchmark response",
    )
    openai_constructor = mocker.patch("openai.OpenAI", return_value=openai_client)

    client = LLMClient(model="benchmark-model")
    result = client.generate_text("Summarize this invoice")

    assert result == "The benchmark response"
    openai_constructor.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://sumopod.test/v1",
    )
    openai_client.chat.completions.create.assert_called_once_with(
        model="benchmark-model",
        messages=[{"role": "user", "content": "Summarize this invoice"}],
    )


def test_llm_client_raises_custom_auth_error_for_invalid_credentials(
    mocker,
    openai_env,
):
    auth_error = _openai_api_error(openai.AuthenticationError, status_code=401)
    openai_client = mocker.Mock()
    openai_client.chat.completions.create.side_effect = auth_error
    mocker.patch("openai.OpenAI", return_value=openai_client)

    client = LLMClient(model="benchmark-model")

    with pytest.raises(LLMAuthError, match="Authentication failed"):
        client.generate_text("Summarize this invoice")

    assert openai_client.chat.completions.create.call_count == 1


def test_llm_client_retries_once_after_rate_limit_and_returns_generated_text(
    mocker,
    openai_env,
):
    rate_limit_error = _openai_api_error(openai.RateLimitError, status_code=429)
    successful_response = _chat_completion_response(mocker, "Recovered response")
    openai_client = mocker.Mock()
    openai_client.chat.completions.create.side_effect = [
        rate_limit_error,
        successful_response,
    ]
    mocker.patch("openai.OpenAI", return_value=openai_client)
    sleep_mock = mocker.patch("time.sleep")

    client = LLMClient(
        model="benchmark-model",
        max_retries=1,
        retry_delay_seconds=2,
    )
    result = client.generate_text("Summarize this invoice")

    assert result == "Recovered response"
    assert openai_client.chat.completions.create.call_count == 2
    sleep_mock.assert_called_once_with(2)
