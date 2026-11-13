# Copyright (c) Microsoft. All rights reserved.

import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import voyageai.error
from agent_framework.exceptions import (
    IntegrationException,
    IntegrationInvalidAuthException,
    IntegrationInvalidRequestException,
    IntegrationInvalidResponseException,
    SettingNotFoundError,
)
from voyageai.client_async import AsyncClient

from agent_framework_voyageai import RawVoyageAIEmbeddingClient, VoyageAIEmbeddingClient
from agent_framework_voyageai._embedding_client import MAX_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]

# region: helpers


@pytest.fixture(autouse=True)
def _clear_voyage_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's real Voyage AI settings out of the unit tests.

    Integration tests keep the environment, since it holds their credentials.
    """
    if request.node.get_closest_marker("integration"):
        return
    for name in ("VOYAGE_API_KEY", "VOYAGE_EMBEDDING_MODEL", "VOYAGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def make_response(vectors: list[list[float]], total_tokens: int | None = 4) -> Any:
    """Shape of ``voyageai.object.EmbeddingsObject``; ``None`` leaves the token count off."""
    response = SimpleNamespace(embeddings=vectors)
    if total_tokens is not None:
        response.total_tokens = total_tokens
    return response


def make_client(*responses: Any, model: str | None = "voyage-4") -> tuple[VoyageAIEmbeddingClient, AsyncMock]:
    sdk = AsyncMock(spec=AsyncClient)
    sdk.embed.side_effect = list(responses)
    return VoyageAIEmbeddingClient(model=model, client=sdk), sdk


# endregion

# region: the model is always explicit


def test_model_is_required() -> None:
    with pytest.raises(SettingNotFoundError, match="embedding_model"):
        VoyageAIEmbeddingClient(api_key="key")


@pytest.mark.parametrize("model", ["mistral-embed", "text-embedding-3-small", "voyage-4-nano", "voyage-99"])
def test_unknown_or_other_provider_model_is_rejected_on_construction(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Voyage AI embedding model"):
        VoyageAIEmbeddingClient(model=model, api_key="key")


async def test_per_call_model_is_validated_before_any_request() -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="Unsupported Voyage AI embedding model"):
        await client.get_embeddings(["a"], options={"model": "rerank-3"})
    sdk.embed.assert_not_awaited()


def test_model_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "voyage-3.5-lite")
    assert RawVoyageAIEmbeddingClient().model == "voyage-3.5-lite"


def test_api_key_is_required_without_a_client() -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageAIEmbeddingClient(model="voyage-4")


def test_client_must_be_a_voyage_client() -> None:
    with pytest.raises(TypeError, match="AsyncClient"):
        VoyageAIEmbeddingClient(model="voyage-4", client=cast(Any, object()))


# endregion

# region: request constraints


@pytest.mark.parametrize(
    ("model", "dimensions"),
    [("voyage-law-2", 512), ("voyage-finance-2", 2048), ("voyage-4", 100), ("voyage-code-3", 4096)],
)
async def test_unsupported_dimensions_are_rejected_before_any_request(model: str, dimensions: int) -> None:
    client, sdk = make_client(model=model)
    with pytest.raises(ValueError, match="supports dimensions"):
        await client.get_embeddings(["a"], options={"dimensions": dimensions})
    sdk.embed.assert_not_awaited()


@pytest.mark.parametrize("input_type", ["passage", "search_query", ""])
async def test_input_type_must_be_document_or_query(input_type: str) -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="input_type"):
        await client.get_embeddings(["a"], options=cast(Any, {"input_type": input_type}))
    sdk.embed.assert_not_awaited()


async def test_optional_parameters_are_sent_only_when_requested() -> None:
    client, sdk = make_client(make_response([[0.1, 0.2]]), make_response([[0.1] * 256]))

    await client.get_embeddings(["a"])
    assert sdk.embed.await_args_list[0].kwargs == {"model": "voyage-4"}

    await client.get_embeddings(["a"], options={"dimensions": 256, "input_type": "query", "truncation": False})
    assert sdk.embed.await_args_list[1].kwargs == {
        "model": "voyage-4",
        "input_type": "query",
        "truncation": False,
        "output_dimension": 256,
    }


async def test_truncation_must_be_a_bool() -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="truncation"):
        await client.get_embeddings(["a"], options=cast(Any, {"truncation": "yes"}))
    sdk.embed.assert_not_awaited()


async def test_empty_input_sends_nothing() -> None:
    client, sdk = make_client()
    assert list(await client.get_embeddings([])) == []
    sdk.embed.assert_not_awaited()


async def test_inputs_over_the_request_limit_are_split_in_order_and_usage_is_summed() -> None:
    values = [str(i) for i in range(MAX_BATCH_SIZE + 1)]
    client, sdk = make_client(
        make_response([[float(i)] for i in range(MAX_BATCH_SIZE)], total_tokens=10),
        make_response([[float(MAX_BATCH_SIZE)]], total_tokens=3),
    )

    result = await client.get_embeddings(values)

    assert [len(call.args[0]) for call in sdk.embed.await_args_list] == [MAX_BATCH_SIZE, 1]
    assert [embedding.vector[0] for embedding in result] == [float(i) for i in range(MAX_BATCH_SIZE + 1)]
    assert result.usage == {"input_token_count": 13, "total_token_count": 13}


# endregion

# region: the response is checked, and billing fails closed


async def test_missing_token_count_fails_instead_of_reporting_zero_tokens() -> None:
    client, _ = make_client(make_response([[0.1, 0.2]], total_tokens=None))
    with pytest.raises(IntegrationInvalidResponseException, match="token count"):
        await client.get_embeddings(["a"])


@pytest.mark.parametrize("total_tokens", [-1, "4", True, 1.5])
async def test_invalid_token_count_fails(total_tokens: Any) -> None:
    client, _ = make_client(make_response([[0.1, 0.2]], total_tokens=total_tokens))
    with pytest.raises(IntegrationInvalidResponseException, match="token count"):
        await client.get_embeddings(["a"])


async def test_embedding_count_must_match_input_count() -> None:
    client, _ = make_client(make_response([[0.1, 0.2]]))
    with pytest.raises(IntegrationInvalidResponseException, match="expected 2 embeddings"):
        await client.get_embeddings(["a", "b"])


async def test_embedding_length_must_match_requested_dimensions() -> None:
    client, _ = make_client(make_response([[0.1, 0.2]]))
    with pytest.raises(IntegrationInvalidResponseException, match="unexpected embedding length"):
        await client.get_embeddings(["a"], options={"dimensions": 256})


# endregion

# region: error mapping


def _error(cls: type[voyageai.error.VoyageError], status: int | None, detail: str) -> voyageai.error.VoyageError:
    return cls(detail, http_body=f'{{"detail": "{detail}"}}', http_status=status, json_body={"detail": detail})


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            _error(voyageai.error.AuthenticationError, 401, "Provided API key is invalid."),
            IntegrationInvalidAuthException,
        ),
        (_error(voyageai.error.APIError, 403, "Forbidden."), IntegrationInvalidAuthException),
        (_error(voyageai.error.InvalidRequestError, 400, "bad input"), IntegrationInvalidRequestException),
        (_error(voyageai.error.MalformedRequestError, 422, "malformed"), IntegrationInvalidRequestException),
        (_error(voyageai.error.RateLimitError, 429, "slow down"), IntegrationException),
        (_error(voyageai.error.ServerError, 500, "boom"), IntegrationException),
        (_error(voyageai.error.APIConnectionError, None, "no route"), IntegrationException),
    ],
)
async def test_sdk_errors_map_to_framework_exceptions_with_the_detail(
    error: voyageai.error.VoyageError, expected: type[Exception]
) -> None:
    client, sdk = make_client()
    sdk.embed.side_effect = error

    with pytest.raises(expected) as exc_info:
        await client.get_embeddings(["a"])

    assert type(exc_info.value) is expected
    assert str(error.user_message) in str(exc_info.value)


async def test_detail_comes_from_the_error_body() -> None:
    client, sdk = make_client()
    error = voyageai.error.InvalidRequestError(
        "sdk message", http_body="{}", http_status=400, json_body={"detail": "Value 'x' is not valid."}
    )
    sdk.embed.side_effect = error

    with pytest.raises(IntegrationInvalidRequestException, match="Value 'x' is not valid."):
        await client.get_embeddings(["a"])


async def test_error_without_a_detail_body_falls_back_to_the_sdk_message() -> None:
    client, sdk = make_client()
    sdk.embed.side_effect = voyageai.error.InvalidRequestError("sdk message", http_status=400)
    with pytest.raises(IntegrationInvalidRequestException, match="sdk message"):
        await client.get_embeddings(["a"])


async def test_unexpected_sdk_failure_is_wrapped() -> None:
    client, sdk = make_client()
    sdk.embed.side_effect = RuntimeError("connection reset")
    with pytest.raises(IntegrationException, match="connection reset"):
        await client.get_embeddings(["a"])


# endregion

# region: integration


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY is not set")
async def test_integration_embeds_with_requested_dimensions() -> None:
    client = VoyageAIEmbeddingClient(model="voyage-4-lite")

    result = await client.get_embeddings(["hello"], options={"dimensions": 256, "input_type": "query"})

    assert len(result) == 1
    assert result[0].dimensions == 256
    assert result.usage is not None
    assert (result.usage.get("total_token_count") or 0) >= 0


# endregion
