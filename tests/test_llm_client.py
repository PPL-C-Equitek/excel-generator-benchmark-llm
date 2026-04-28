import httpx
import openai
import pytest

from src.llm_client import LLMAuthError, LLMClient


def _gateway_api_error(error_cls, status_code):
    request = httpx.Request(
        "POST",
        "https://sumopod.test/v1/chat/completions",
    )
    response = httpx.Response(status_code=status_code, request=request)
    return error_cls(
        f"LLM gateway returned HTTP {status_code}",
        response=response,
        body={"error": {"message": f"HTTP {status_code}"}},
    )


def _chat_completion_response(mocker, content):
    message = mocker.Mock(content=content)
    choice = mocker.Mock(message=message)
    return mocker.Mock(choices=[choice])


@pytest.fixture
def gateway_env(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("BASE_URL", "https://sumopod.test/v1")


def test_llm_client_initializes_gateway_and_returns_generated_text(
    mocker,
    gateway_env,
):
    gateway_client = mocker.Mock()
    gateway_client.chat.completions.create.return_value = _chat_completion_response(
        mocker,
        "The benchmark response",
    )
    sdk_constructor = mocker.patch("openai.OpenAI", return_value=gateway_client)

    client = LLMClient(model="benchmark-model")
    result = client.generate_text("Summarize this invoice")

    assert result == "The benchmark response"
    sdk_constructor.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://sumopod.test/v1",
    )
    gateway_client.chat.completions.create.assert_called_once_with(
        model="benchmark-model",
        messages=[{"role": "user", "content": "Summarize this invoice"}],
    )


def test_llm_client_raises_custom_auth_error_for_invalid_credentials(
    mocker,
    gateway_env,
):
    auth_error = _gateway_api_error(openai.AuthenticationError, status_code=401)
    gateway_client = mocker.Mock()
    gateway_client.chat.completions.create.side_effect = auth_error
    mocker.patch("openai.OpenAI", return_value=gateway_client)

    client = LLMClient(model="benchmark-model")

    with pytest.raises(LLMAuthError, match="Authentication failed"):
        client.generate_text("Summarize this invoice")

    assert gateway_client.chat.completions.create.call_count == 1


def test_llm_client_retries_once_after_rate_limit_and_returns_generated_text(
    mocker,
    gateway_env,
):
    rate_limit_error = _gateway_api_error(openai.RateLimitError, status_code=429)
    successful_response = _chat_completion_response(mocker, "Recovered response")
    gateway_client = mocker.Mock()
    gateway_client.chat.completions.create.side_effect = [
        rate_limit_error,
        successful_response,
    ]
    mocker.patch("openai.OpenAI", return_value=gateway_client)
    sleep_mock = mocker.patch("time.sleep")

    client = LLMClient(
        model="benchmark-model",
        max_retries=1,
        retry_delay_seconds=2,
    )
    result = client.generate_text("Summarize this invoice")

    assert result == "Recovered response"
    assert gateway_client.chat.completions.create.call_count == 2
    sleep_mock.assert_called_once_with(2)


def test_llm_client_raises_rate_limit_error_when_retries_are_exhausted(
    mocker,
    gateway_env,
):
    rate_limit_error = _gateway_api_error(openai.RateLimitError, status_code=429)
    gateway_client = mocker.Mock()
    gateway_client.chat.completions.create.side_effect = rate_limit_error
    mocker.patch("openai.OpenAI", return_value=gateway_client)
    sleep_mock = mocker.patch("time.sleep")

    client = LLMClient(
        model="benchmark-model",
        max_retries=0,
        retry_delay_seconds=2,
    )

    with pytest.raises(openai.RateLimitError):
        client.generate_text("Summarize this invoice")

    assert gateway_client.chat.completions.create.call_count == 1
    sleep_mock.assert_not_called()


def test_llm_client_raises_runtime_error_when_no_attempts_are_configured(
    mocker,
    gateway_env,
):
    gateway_client = mocker.Mock()
    mocker.patch("openai.OpenAI", return_value=gateway_client)

    client = LLMClient(model="benchmark-model", max_retries=-1)

    with pytest.raises(RuntimeError, match="failed unexpectedly"):
        client.generate_text("Summarize this invoice")

    gateway_client.chat.completions.create.assert_not_called()
