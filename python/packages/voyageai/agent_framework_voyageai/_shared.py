# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import importlib.metadata
from collections.abc import Sequence
from typing import Any, Final, NoReturn, cast

import voyageai.error
from agent_framework.exceptions import (
    IntegrationException,
    IntegrationInvalidAuthException,
    IntegrationInvalidRequestException,
    IntegrationInvalidResponseException,
)
from voyageai.client_async import AsyncClient

VOYAGEAI_API_BASE_URL: Final = "https://api.voyageai.com/v1"
DEFAULT_TIMEOUT_SECONDS: Final = 60.0
MAX_ERROR_DETAIL_CHARS: Final = 2000

MAX_BATCH_SIZE: Final = 1000
"""The most inputs Voyage AI accepts in one embeddings request."""

MATRYOSHKA_DIMENSIONS: Final = (256, 512, 1024, 2048)
INPUT_TYPES: Final = ("document", "query")

# Voyage AI tokenizers produced at most one token per UTF-8 byte of text in our checks, plus a couple
# of special tokens per input. This bound adds room, and lets a request that is far below its token
# ceiling skip the tokenizer, which is downloaded from the Hugging Face Hub on first use.
_TOKENS_PER_INPUT_OVERHEAD: Final = 8


def error_detail(ex: voyageai.error.VoyageError) -> str:
    """Return the message of a Voyage AI error.

    Voyage AI error bodies are ``{"detail": "<text>"}``. Anything else falls back to the SDK message.
    """
    body: Any = getattr(ex, "json_body", None)
    detail: Any = cast("dict[str, Any]", body).get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        detail = str(getattr(ex, "user_message", None) or ex)
    return detail[:MAX_ERROR_DETAIL_CHARS]


def raise_sdk_error(ex: voyageai.error.VoyageError, operation: str) -> NoReturn:
    """Raise the framework exception for a Voyage AI SDK error.

    ``401``/``403`` are authentication errors, other ``4xx`` except ``429`` are invalid requests, and
    everything else, including ``429`` and ``5xx``, is a plain integration failure.
    """
    status: int | None = getattr(ex, "http_status", None)
    if status is not None and status < 400:
        raise IntegrationInvalidResponseException(
            f"Voyage AI {operation} response was invalid: {error_detail(ex)}",
            inner_exception=ex,
        ) from ex

    message = f"Voyage AI {operation} request failed with status {status}: {error_detail(ex)}"
    if isinstance(ex, voyageai.error.AuthenticationError) or status in (401, 403):
        raise IntegrationInvalidAuthException(message) from ex
    if status is not None and 400 <= status < 500 and status != 429:
        raise IntegrationInvalidRequestException(message) from ex
    raise IntegrationException(message, inner_exception=ex) from ex


def create_sdk_client(api_key: str, base_url: str | None) -> AsyncClient:
    """Create the SDK client and record this package as the integration wrapper."""
    client = AsyncClient(api_key=api_key, timeout=DEFAULT_TIMEOUT_SECONDS, base_url=base_url)
    try:
        package_version = importlib.metadata.version("agent-framework-voyageai")
    except importlib.metadata.PackageNotFoundError:
        package_version = "0.0.0"
    client.append_client_metadata("agent-framework-voyageai", package_version)
    return client


def validate_input_type(input_type: Any, allowed: Sequence[str] = INPUT_TYPES) -> None:
    if input_type is not None and input_type not in allowed:
        raise ValueError(f"Unsupported Voyage AI input_type {input_type!r}; use one of {list(allowed)}.")


def validate_dimensions(model: str, dimensions: Any, supported: Sequence[int]) -> None:
    if dimensions is not None and (type(dimensions) is not int or dimensions not in supported):
        raise ValueError(f"Model {model!r} supports dimensions {list(supported)}, not {dimensions!r}.")


async def split_into_requests(
    client: AsyncClient,
    model: str,
    values: Sequence[str],
    *,
    max_request_tokens: int | None,
) -> list[Sequence[str]]:
    """Split ``values`` into consecutive requests within Voyage AI's per-request limits.

    A request holds at most ``MAX_BATCH_SIZE`` inputs and, for models with a documented ceiling,
    at most ``max_request_tokens`` tokens in total. Tokens are counted with the model's own tokenizer
    through the SDK, unless the inputs are small enough to be under the ceiling whatever they tokenize
    to. An input over the ceiling on its own still gets its own request, and the service decides.

    Raises:
        IntegrationException: If the tokenizer is needed and cannot be loaded. Requests are never sent
            unsplit in that case.
    """
    if max_request_tokens is None or (
        sum(len(value.encode("utf-8")) + _TOKENS_PER_INPUT_OVERHEAD for value in values) <= max_request_tokens
    ):
        return [values[start : start + MAX_BATCH_SIZE] for start in range(0, len(values), MAX_BATCH_SIZE)]

    try:
        encodings = await asyncio.to_thread(client.tokenize, list(values), model)
        token_counts = [len(encoding) for encoding in encodings]
    except Exception as ex:
        raise IntegrationException(
            f"Voyage AI request could not be split by token count: the tokenizer for {model!r} failed: {ex}",
            inner_exception=ex,
        ) from ex

    requests: list[Sequence[str]] = []
    start, tokens = 0, 0
    for index, count in enumerate(token_counts):
        if index > start and (index - start == MAX_BATCH_SIZE or tokens + count > max_request_tokens):
            requests.append(values[start:index])
            start, tokens = index, 0
        tokens += count
    requests.append(values[start:])
    return requests


def invalid_response(ex: Exception, operation: str) -> IntegrationInvalidResponseException:
    return IntegrationInvalidResponseException(
        f"Voyage AI {operation} response was invalid: {ex}",
        inner_exception=ex,
    )
