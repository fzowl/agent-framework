# Copyright (c) Microsoft. All rights reserved.

import os
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import voyageai
import voyageai.error
from agent_framework.exceptions import (
    IntegrationException,
    IntegrationInvalidAuthException,
    IntegrationInvalidRequestException,
    IntegrationInvalidResponseException,
    SettingNotFoundError,
)
from voyageai.api_resources import VoyageResponse
from voyageai.client_async import AsyncClient
from voyageai.object import EmbeddingsObject

from agent_framework_voyageai import RawVoyageAIEmbeddingClient, VoyageAIEmbeddingClient
from agent_framework_voyageai._shared import MAX_BATCH_SIZE  # pyright: ignore[reportPrivateUsage]

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


def make_body(vectors: list[list[float]], usage: Any = None) -> dict[str, Any]:
    """The JSON body of ``POST /v1/embeddings``."""
    return {
        "object": "list",
        "model": "voyage-4",
        "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)],
        "usage": {"total_tokens": 4} if usage is None else usage,
    }


def make_response(vectors: list[list[float]], total_tokens: int = 4) -> EmbeddingsObject:
    """What ``AsyncClient.embed`` returns for a well-formed body."""
    return EmbeddingsObject(VoyageResponse.construct_from(make_body(vectors, {"total_tokens": total_tokens})))


def make_client(*responses: Any, model: str | None = "voyage-4") -> tuple[VoyageAIEmbeddingClient, AsyncMock]:
    sdk = AsyncMock(spec=AsyncClient)
    sdk.count_tokens.return_value = 1
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


@pytest.mark.parametrize("api_key", ["", "  "])
def test_blank_api_key_is_rejected(api_key: str) -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageAIEmbeddingClient(model="voyage-4", api_key=api_key)


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


@pytest.mark.parametrize("dimensions", [1024.0, True])
async def test_dimensions_must_be_an_integer(dimensions: Any) -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="supports dimensions"):
        await client.get_embeddings(["a"], options=cast(Any, {"dimensions": dimensions}))
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


async def test_requests_stay_within_the_models_token_ceiling() -> None:
    client, sdk = make_client(make_response([[0.1]] * 2), make_response([[0.1]]), model="voyage-4-large")
    sdk.count_tokens.side_effect = [50_000, 50_000, 50_000]

    await client.get_embeddings(["a" * 50_000] * 3)

    assert [len(call.args[0]) for call in sdk.embed.await_args_list] == [2, 1]


async def test_input_over_the_ceiling_on_its_own_gets_its_own_request() -> None:
    client, sdk = make_client(make_response([[0.1]]), make_response([[0.1]]), model="voyage-4-large")
    sdk.count_tokens.side_effect = [130_000, 10]

    await client.get_embeddings(["a" * 130_000, "b" * 10_000])

    assert [len(call.args[0]) for call in sdk.embed.await_args_list] == [1, 1]


async def test_tokens_are_counted_for_every_call_to_a_model_with_a_ceiling() -> None:
    client, sdk = make_client(make_response([[0.1]]))
    sdk.count_tokens.side_effect = [3]

    await client.get_embeddings(["short text"])

    sdk.count_tokens.assert_called_once_with(["short text"], "voyage-4")


async def test_model_without_a_documented_ceiling_is_split_by_input_count_only() -> None:
    client, sdk = make_client(make_response([[0.1]]), model="voyage-code-4")
    await client.get_embeddings(["a" * 500_000])
    sdk.count_tokens.assert_not_called()


async def test_tokenizer_failure_fails_the_call_instead_of_sending_unsplit_requests() -> None:
    client, sdk = make_client(model="voyage-4-large")
    sdk.count_tokens.side_effect = OSError("hub unreachable")

    with pytest.raises(IntegrationException, match="could not be split by token count"):
        await client.get_embeddings(["a" * 150_000])

    sdk.embed.assert_not_awaited()


# endregion

# region: the response is checked, and billing fails closed


@pytest.mark.parametrize(
    "usage_body",
    [
        pytest.param({"drop_usage": True}, id="no-usage"),
        pytest.param({}, id="empty-usage"),
        pytest.param({"total_tokens": None}, id="null-total-tokens"),
    ],
)
async def test_response_without_a_token_count_fails_instead_of_reporting_zero_tokens(
    monkeypatch: pytest.MonkeyPatch, usage_body: dict[str, Any]
) -> None:
    body = make_body([[0.1, 0.2]])
    if usage_body.get("drop_usage"):
        del body["usage"]
    else:
        body["usage"] = usage_body
    # Run the SDK's own response handling, which is where the token count is read.
    monkeypatch.setattr(voyageai.Embedding, "acreate", AsyncMock(return_value=VoyageResponse.construct_from(body)))
    client = VoyageAIEmbeddingClient(model="voyage-4", client=AsyncClient(api_key="test-key"))

    with pytest.raises(IntegrationInvalidResponseException):
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


async def test_non_text_detail_does_not_escape_as_a_type_error() -> None:
    client, sdk = make_client()
    sdk.embed.side_effect = voyageai.error.InvalidRequestError(
        "sdk message", http_status=422, json_body={"detail": [{"msg": "field required"}]}
    )
    with pytest.raises(IntegrationInvalidRequestException, match="sdk message"):
        await client.get_embeddings(["a"])


async def test_sdk_error_without_a_status_for_an_inconsistent_body_is_an_invalid_response() -> None:
    client, sdk = make_client()
    sdk.embed.side_effect = voyageai.error.ServerError("body was inconsistent")
    with pytest.raises(IntegrationInvalidResponseException, match="body was inconsistent"):
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

    result = await client.get_embeddings(
        ["Voyage AI provides embedding models."], options={"dimensions": 256, "input_type": "document"}
    )

    assert len(result) == 1
    assert result[0].dimensions == 256
    assert result.usage is not None
    assert (result.usage.get("total_token_count") or 0) > 0


# endregion
