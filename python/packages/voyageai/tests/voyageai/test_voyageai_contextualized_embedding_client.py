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
from voyageai.object import ContextualizedEmbeddingsObject

from agent_framework_voyageai import VoyageAIContextualizedEmbeddingClient

# region: helpers


@pytest.fixture(autouse=True)
def _clear_voyage_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("integration"):
        return
    for name in ("VOYAGE_API_KEY", "VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL", "VOYAGE_BASE_URL"):
        monkeypatch.delenv(name, raising=False)


def make_body(documents: list[list[list[float]]], texts: list[list[str]] | None = None, usage: Any = None) -> Any:
    """The JSON body of ``POST /v1/contextualizedembeddings``; ``texts`` are the chunk texts, if returned."""
    data: list[dict[str, Any]] = []
    for i, vectors in enumerate(documents):
        chunks: list[dict[str, Any]] = []
        for j, vector in enumerate(vectors):
            chunk: dict[str, Any] = {"object": "embedding", "embedding": vector, "index": j}
            if texts is not None:
                chunk["text"] = texts[i][j]
            chunks.append(chunk)
        data.append({"object": "list", "data": chunks, "index": i})
    return {
        "object": "list",
        "data": data,
        "model": "voyage-context-4",
        "usage": {"total_tokens": 7} if usage is None else usage,
    }


def make_response(
    documents: list[list[list[float]]], texts: list[list[str]] | None = None, total_tokens: int = 7
) -> ContextualizedEmbeddingsObject:
    body = make_body(documents, texts, {"total_tokens": total_tokens})
    return ContextualizedEmbeddingsObject(VoyageResponse.construct_from(body))


def make_client(
    *responses: Any, model: str | None = "voyage-context-4"
) -> tuple[VoyageAIContextualizedEmbeddingClient, AsyncMock]:
    sdk = AsyncMock(spec=AsyncClient)
    sdk.contextualized_embed.side_effect = list(responses)
    return VoyageAIContextualizedEmbeddingClient(model=model, client=sdk), sdk


# endregion

# region: the model is always explicit


def test_model_is_required() -> None:
    with pytest.raises(SettingNotFoundError, match="contextualized_embedding_model"):
        VoyageAIContextualizedEmbeddingClient(api_key="key")


@pytest.mark.parametrize("model", ["voyage-4", "mistral-embed", "voyage-context-2", "rerank-3"])
def test_other_models_are_rejected_on_construction(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Voyage AI contextualized embedding model"):
        VoyageAIContextualizedEmbeddingClient(model=model, api_key="key")


async def test_per_call_model_is_validated_before_any_request() -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="Unsupported Voyage AI contextualized embedding model"):
        await client.get_contextualized_embeddings(["a"], options={"model": "voyage-4", "input_type": "document"})
    sdk.contextualized_embed.assert_not_awaited()


def test_model_can_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "key")
    monkeypatch.setenv("VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL", "voyage-context-3")
    assert VoyageAIContextualizedEmbeddingClient().model == "voyage-context-3"


@pytest.mark.parametrize("api_key", [None, "", "  "])
def test_api_key_is_required_without_a_client(api_key: str | None) -> None:
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageAIContextualizedEmbeddingClient(model="voyage-context-4", api_key=api_key)


# endregion

# region: the API constraints


async def test_input_type_is_required() -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="input_type is required"):
        await client.get_contextualized_embeddings(["a"], options={})
    sdk.contextualized_embed.assert_not_awaited()


@pytest.mark.parametrize("input_type", ["passage", ""])
async def test_input_type_must_be_document_or_query(input_type: str) -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="input_type"):
        await client.get_contextualized_embeddings(["a"], options=cast(Any, {"input_type": input_type}))
    sdk.contextualized_embed.assert_not_awaited()


@pytest.mark.parametrize("dimensions", [100, 4096, 1024.0, True])
async def test_unsupported_dimensions_are_rejected_before_any_request(dimensions: Any) -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="supports dimensions"):
        await client.get_contextualized_embeddings(
            ["a"], options=cast(Any, {"input_type": "document", "dimensions": dimensions})
        )
    sdk.contextualized_embed.assert_not_awaited()


async def test_pre_chunked_documents_are_rejected_before_any_request() -> None:
    client, sdk = make_client()
    with pytest.raises(ValueError, match="send each document as one string"):
        await client.get_contextualized_embeddings(
            cast(Any, [["chunk one", "chunk two"]]), options={"input_type": "document"}
        )
    sdk.contextualized_embed.assert_not_awaited()


async def test_documents_are_sent_flat_with_auto_chunking_at_32000() -> None:
    client, sdk = make_client(make_response([[[0.1, 0.2]], [[0.3, 0.4]]]))

    await client.get_contextualized_embeddings(["doc one", "doc two"], options={"input_type": "document"})

    call = sdk.contextualized_embed.await_args
    assert call is not None
    assert call.args == (["doc one", "doc two"],)
    assert call.kwargs == {
        "model": "voyage-context-4",
        "input_type": "document",
        "enable_auto_chunking": True,
        "chunk_size": 32_000,
    }


async def test_queries_are_sent_unchunked_one_per_inner_list() -> None:
    client, sdk = make_client(make_response([[[0.1] * 256], [[0.2] * 256]]))

    await client.get_contextualized_embeddings(["q1", "q2"], options={"input_type": "query", "dimensions": 256})

    call = sdk.contextualized_embed.await_args
    assert call is not None
    assert call.args == ([["q1"], ["q2"]],)
    assert call.kwargs == {"model": "voyage-context-4", "input_type": "query", "output_dimension": 256}


async def test_empty_input_sends_nothing() -> None:
    client, sdk = make_client()
    assert list(await client.get_contextualized_embeddings([], options={"input_type": "document"})) == []
    sdk.contextualized_embed.assert_not_awaited()


# endregion

# region: results


async def test_chunks_are_returned_with_their_vectors_when_the_service_returns_them() -> None:
    client, _ = make_client(
        make_response([[[0.1], [0.2]], [[0.3]]], texts=[["first half", "second half"], ["whole"]], total_tokens=9)
    )

    result = await client.get_contextualized_embeddings(["a long doc", "short"], options={"input_type": "document"})

    assert [doc.chunks for doc in result] == [["first half", "second half"], ["whole"]]
    assert [[e.vector for e in doc.embeddings] for doc in result] == [[[0.1], [0.2]], [[0.3]]]
    assert result.usage == {"input_token_count": 9, "total_token_count": 9}


async def test_chunks_are_none_when_the_service_returns_no_text() -> None:
    client, _ = make_client(make_response([[[0.1]]]))
    result = await client.get_contextualized_embeddings(["q"], options={"input_type": "query"})
    assert result[0].chunks is None


async def test_result_count_must_match_input_count() -> None:
    client, _ = make_client(make_response([[[0.1]]]))
    with pytest.raises(IntegrationInvalidResponseException, match="expected 2 results"):
        await client.get_contextualized_embeddings(["a", "b"], options={"input_type": "document"})


async def test_embedding_length_must_match_requested_dimensions() -> None:
    client, _ = make_client(make_response([[[0.1, 0.2]]]))
    with pytest.raises(IntegrationInvalidResponseException, match="unexpected embedding length"):
        await client.get_contextualized_embeddings(["a"], options={"input_type": "document", "dimensions": 256})


@pytest.mark.parametrize(
    "usage",
    [
        pytest.param("drop", id="no-usage"),
        pytest.param({}, id="empty-usage"),
        pytest.param({"total_tokens": None}, id="null"),
    ],
)
async def test_response_without_a_token_count_fails_instead_of_reporting_zero_tokens(
    monkeypatch: pytest.MonkeyPatch, usage: Any
) -> None:
    body = make_body([[[0.1]]])
    if usage == "drop":
        del body["usage"]
    else:
        body["usage"] = usage
    monkeypatch.setattr(
        voyageai.ContextualizedEmbedding, "acreate", AsyncMock(return_value=VoyageResponse.construct_from(body))
    )
    client = VoyageAIContextualizedEmbeddingClient(model="voyage-context-4", client=AsyncClient(api_key="test-key"))

    with pytest.raises(IntegrationInvalidResponseException):
        await client.get_contextualized_embeddings(["a"], options={"input_type": "document"})


async def test_requests_stay_within_the_token_ceiling_and_usage_is_summed() -> None:
    client, sdk = make_client(
        make_response([[[0.1]], [[0.2]]], total_tokens=5), make_response([[[0.3]]], total_tokens=2)
    )
    sdk.tokenize.return_value = [[0] * 50_000, [0] * 50_000, [0] * 50_000]

    result = await client.get_contextualized_embeddings(["a" * 50_000] * 3, options={"input_type": "document"})

    assert [len(call.args[0]) for call in sdk.contextualized_embed.await_args_list] == [2, 1]
    assert [doc.embeddings[0].vector for doc in result] == [[0.1], [0.2], [0.3]]
    assert result.usage == {"input_token_count": 7, "total_token_count": 7}


# endregion

# region: error mapping


@pytest.mark.parametrize(
    ("status", "error_class", "expected"),
    [
        (401, voyageai.error.AuthenticationError, IntegrationInvalidAuthException),
        (400, voyageai.error.InvalidRequestError, IntegrationInvalidRequestException),
        (429, voyageai.error.RateLimitError, IntegrationException),
        (500, voyageai.error.ServerError, IntegrationException),
    ],
)
async def test_sdk_errors_map_to_framework_exceptions_with_the_detail(
    status: int, error_class: type[voyageai.error.VoyageError], expected: type[Exception]
) -> None:
    client, sdk = make_client()
    sdk.contextualized_embed.side_effect = error_class(
        "sdk message", http_status=status, json_body={"detail": "the detail"}
    )

    with pytest.raises(expected, match="the detail") as exc_info:
        await client.get_contextualized_embeddings(["a"], options={"input_type": "document"})

    assert type(exc_info.value) is expected


# endregion

# region: integration


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("VOYAGE_API_KEY"), reason="VOYAGE_API_KEY is not set")
@pytest.mark.parametrize("model", ["voyage-context-4", "voyage-context-3"])
async def test_integration_documents_and_queries(model: str) -> None:
    client = VoyageAIContextualizedEmbeddingClient(model=model)

    documents = await client.get_contextualized_embeddings(
        ["Cats purr when they are content."], options={"input_type": "document", "dimensions": 256}
    )
    queries = await client.get_contextualized_embeddings(
        ["Why do cats purr?"], options={"input_type": "query", "dimensions": 256}
    )

    assert documents[0].chunks == ["Cats purr when they are content."]
    assert documents[0].embeddings[0].dimensions == 256
    assert queries[0].chunks is None
    assert ((documents.usage or {}).get("total_token_count") or 0) > 0


# endregion
