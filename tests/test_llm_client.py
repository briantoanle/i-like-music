"""Tests for the LM Studio client.

Note: these tests hit a real local server. They are skipped when LM Studio
is not running (connection refused).
"""

import pytest
from unittest.mock import patch
import urllib.request
import urllib.error


from llm_client import LMStudioClient, LMStudioError


# -- unit tests (no server needed) ------------------------------------------

def test_chat_returns_assistant_text():
    client = LMStudioClient()
    fake_response = {
        "choices": [{"message": {"content": "  Here are some lofi beats.  "}}],
    }
    with patch.object(client, "_post", return_value=fake_response):
        result = client.chat([{"role": "user", "content": "recommend chill music"}])
    assert result == "Here are some lofi beats."


def test_chat_returns_none_on_empty_content():
    client = LMStudioClient()
    fake_response = {"choices": [{"message": {"content": ""}}]}
    with patch.object(client, "_post", return_value=fake_response):
        result = client.chat([{"role": "user", "content": "hello"}])
    assert result is None


def test_chat_returns_none_on_server_error():
    client = LMStudioClient()
    with patch.object(client, "_post", side_effect=LMStudioError("500")):
        result = client.chat([{"role": "user", "content": "hello"}])
    assert result is None


def test_available_models_parses_list():
    client = LMStudioClient()
    fake_data = [{"id": "llama-3.1-8b"}, {"id": "mistral-7b"}]
    with patch.object(client, "_get", return_value=fake_data):
        models = client.available_models()
    assert models == ["llama-3.1-8b", "mistral-7b"]


def test_connection_error_raised():
    import llm_client
    with patch.object(llm_client.urllib.request, "urlopen", side_effect=urllib.error.URLError("conn refused")):
        with pytest.raises(LMStudioError, match="Cannot connect"):
            client = LMStudioClient()
            client._request("POST", "/v1/chat/completions", {"messages": []})


def test_custom_model_passed_in_body():
    mock_post = patch.object(LMStudioClient, "_post", return_value={"choices": [{"message": {"content": "ok"}}]})
    with mock_post as patched:
        client = LMStudioClient(model="my-local-model")
        client.chat([{"role": "user", "content": "hi"}])
    body = patched.call_args[0][1]
    assert body["model"] == "my-local-model"


def test_extra_kwargs_forwarded():
    mock_post = patch.object(LMStudioClient, "_post", return_value={"choices": [{"message": {"content": "ok"}}]})
    with mock_post as patched:
        client = LMStudioClient()
        client.chat([{"role": "user", "content": "hi"}], temperature=0.7, max_tokens=50)
    body = patched.call_args[0][1]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 50
