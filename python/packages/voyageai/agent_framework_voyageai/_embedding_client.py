# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import importlib.metadata
import logging
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, ClassVar, Final, Generic, Literal, NoReturn, TypedDict, cast

import voyageai.error
from agent_framework import (
    BaseEmbeddingClient,
    Embedding,
    EmbeddingGenerationOptions,
    GeneratedEmbeddings,
    UsageDetails,
    add_usage_details,
    load_settings,
)
from agent_framework._settings import SecretString
from agent_framework._telemetry import mark_feature_used
from agent_framework.exceptions import (
    IntegrationException,
    IntegrationInvalidAuthException,
    IntegrationInvalidRequestException,
    IntegrationInvalidResponseException,
)
from agent_framework.observability import EmbeddingTelemetryLayer
from voyageai.client_async import AsyncClient

from ._feature_usage import FeatureIndex

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

__all__ = [
    "RawVoyageAIEmbeddingClient",
    "VoyageAIEmbeddingClient",
    "VoyageAIEmbeddingOptions",
    "VoyageAIEmbeddingSettings",
]

logger = logging.getLogger("agent_framework.voyageai")

_VOYAGEAI_API_BASE_URL = "https://api.voyageai.com/v1"
_DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_ERROR_DETAIL_CHARS = 2000

MAX_BATCH_SIZE: Final = 1000
"""The most inputs Voyage AI accepts in one embeddings request."""

_MATRYOSHKA_DIMENSIONS: Final = (256, 512, 1024, 2048)
_FIXED_DIMENSIONS: Final = (1024,)

_MODEL_DIMENSIONS: Final[Mapping[str, tuple[int, ...]]] = {
    "voyage-4-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-4-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-4": _MATRYOSHKA_DIMENSIONS,
    "voyage-3-large": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5": _MATRYOSHKA_DIMENSIONS,
    "voyage-3.5-lite": _MATRYOSHKA_DIMENSIONS,
    "voyage-code-3": _MATRYOSHKA_DIMENSIONS,
    "voyage-finance-2": _FIXED_DIMENSIONS,
    "voyage-law-2": _FIXED_DIMENSIONS,
}
"""The text embedding models this client accepts, with the ``dimensions`` each one supports.

The model is always named by the caller; there is no default. A name outside this table is rejected
before any request is sent. The Voyage AI SDK also runs some open-weight models locally, and this
table keeps those from being selected by accident.
"""

_INPUT_TYPES: Final = ("document", "query")


def _batches(values: Sequence[str]) -> Iterator[Sequence[str]]:
    """Split ``values`` into consecutive runs of at most ``MAX_BATCH_SIZE`` inputs."""
    for start in range(0, len(values), MAX_BATCH_SIZE):
        yield values[start : start + MAX_BATCH_SIZE]


class VoyageAIEmbeddingOptions(EmbeddingGenerationOptions, total=False):
    """Voyage AI-specific embedding options.

    Extends EmbeddingGenerationOptions with Voyage AI fields. ``model`` is required, either here
    or on the client. ``dimensions`` is sent as ``output_dimension`` and only when set.

    Keys:
        input_type: ``"document"`` for texts being stored and ``"query"`` for search text. Voyage
            AI prepends a retrieval prompt for either value. Omitted, nothing is prepended and the
            key is not sent.
        truncation: ``True`` shortens over-length input to the model's context window and
            ``False`` rejects it. Omitted, the service default (``True``) applies.

    Examples:
        .. code-block:: python

            from agent_framework_voyageai import VoyageAIEmbeddingOptions

            options: VoyageAIEmbeddingOptions = {
                "model": "voyage-4",
                "dimensions": 512,
                "input_type": "query",
            }
    """

    input_type: Literal["document", "query"]
    truncation: bool


VoyageAIEmbeddingOptionsT = TypeVar(
    "VoyageAIEmbeddingOptionsT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="VoyageAIEmbeddingOptions",
    covariant=True,
)


class VoyageAIEmbeddingSettings(TypedDict, total=False):
    """Voyage AI embedding settings.

    Settings are resolved in this order: explicit keyword arguments, values from an
    explicitly provided .env file, then environment variables with the prefix 'VOYAGE_'.

    Keys:
        api_key: The Voyage AI API key. (Env var VOYAGE_API_KEY)
        embedding_model: The embedding model, for example ``voyage-4``. There is no default.
            (Env var VOYAGE_EMBEDDING_MODEL)
        base_url: Optional endpoint override. (Env var VOYAGE_BASE_URL)
    """

    api_key: SecretString | None
    embedding_model: str | None
    base_url: str | None


def _error_detail(ex: voyageai.error.VoyageError) -> str:
    """Return the message of a Voyage AI error.

    Voyage AI error bodies are ``{"detail": "<text>"}``. Anything else falls back to the SDK message.
    """
    body: Any = getattr(ex, "json_body", None)
    detail: Any = cast("dict[str, Any]", body).get("detail") if isinstance(body, dict) else None
    if not isinstance(detail, str):
        user_message: str | None = getattr(ex, "user_message", None)
        detail = user_message or str(ex)
    return detail[:_MAX_ERROR_DETAIL_CHARS]


def _validate_model_and_dimensions(model: str, dimensions: int | None) -> None:
    supported = _MODEL_DIMENSIONS.get(model)
    if supported is None:
        raise ValueError(
            f"Unsupported Voyage AI embedding model {model!r}. Supported models: {', '.join(_MODEL_DIMENSIONS)}."
        )
    if dimensions is not None and dimensions not in supported:
        raise ValueError(f"Model {model!r} supports dimensions {list(supported)}, not {dimensions!r}.")


class RawVoyageAIEmbeddingClient(
    BaseEmbeddingClient[str, list[float], VoyageAIEmbeddingOptionsT],
    Generic[VoyageAIEmbeddingOptionsT],
):
    """Raw Voyage AI embedding client without telemetry.

    Uses the official ``voyageai`` SDK. The model is never defaulted: pass ``model``, set
    ``VOYAGE_EMBEDDING_MODEL``, or pass ``options["model"]`` on every call.

    Keyword Args:
        model: The Voyage AI embedding model (for example ``"voyage-4"``). Falls back to
            ``VOYAGE_EMBEDDING_MODEL``.
        api_key: The Voyage AI API key. Falls back to ``VOYAGE_API_KEY``.
        base_url: Endpoint override. Falls back to ``VOYAGE_BASE_URL``, then to the Voyage AI
            default. Ignored when ``client`` is provided.
        client: Optional pre-configured ``AsyncClient``. It owns its credentials, so
            ``api_key`` is not required.
        additional_properties: Additional properties stored on the client instance.
        env_file_path: Path to a ``.env`` file for settings.
        env_file_encoding: Encoding of that ``.env`` file.

    Raises:
        SettingNotFoundError: If no model is supplied.
        ValueError: If the model is not a supported Voyage AI text embedding model, or no API key
            is available and no ``client`` was passed.
    """

    INJECTABLE: ClassVar[set[str]] = {"client"}

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretString | None = None,
        base_url: str | None = None,
        client: AsyncClient | None = None,
        additional_properties: dict[str, Any] | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a raw Voyage AI embedding client."""
        if client is not None and not isinstance(client, AsyncClient):
            raise TypeError(f"The 'client' parameter accepts a voyageai AsyncClient; got {type(client).__name__}.")

        settings = load_settings(
            VoyageAIEmbeddingSettings,
            env_prefix="VOYAGE_",
            required_fields=["embedding_model"],
            embedding_model=model,
            api_key=api_key,
            base_url=base_url,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        resolved_model: str = settings["embedding_model"]  # type: ignore[assignment]
        _validate_model_and_dimensions(resolved_model, None)
        self.model: str = resolved_model
        self.base_url: str | None = settings.get("base_url")

        if client is None:
            key = settings.get("api_key")
            if key is None:
                raise ValueError(
                    "A Voyage AI API key is required. Set VOYAGE_API_KEY, pass api_key, "
                    "or supply a pre-configured client."
                )
            client = AsyncClient(
                api_key=key.get_secret_value(),
                timeout=_DEFAULT_TIMEOUT_SECONDS,
                base_url=self.base_url,
            )
            try:
                package_version = importlib.metadata.version("agent-framework-voyageai")
            except importlib.metadata.PackageNotFoundError:
                package_version = "0.0.0"
            client.append_client_metadata("agent-framework-voyageai", package_version)
        self.client = client

        super().__init__(additional_properties=additional_properties)

    def service_url(self) -> str:
        """Get the URL of the service."""
        return self.base_url or _VOYAGEAI_API_BASE_URL

    @staticmethod
    def _raise_sdk_error(ex: voyageai.error.VoyageError) -> NoReturn:
        status: int | None = getattr(ex, "http_status", None)
        if status is not None and status < 400:
            raise IntegrationInvalidResponseException(
                f"Voyage AI embeddings response was invalid: {_error_detail(ex)}",
                inner_exception=ex,
            ) from ex

        message = f"Voyage AI embeddings request failed with status {status}: {_error_detail(ex)}"
        if isinstance(ex, voyageai.error.AuthenticationError) or status in (401, 403):
            raise IntegrationInvalidAuthException(message) from ex
        if status is not None and 400 <= status < 500 and status != 429:
            raise IntegrationInvalidRequestException(message) from ex
        raise IntegrationException(message, inner_exception=ex) from ex

    async def _embed_batch(
        self,
        values: Sequence[str],
        *,
        model: str,
        input_type: str | None,
        truncation: bool | None,
        dimensions: int | None,
    ) -> tuple[list[Embedding[list[float]]], UsageDetails]:
        kwargs: dict[str, Any] = {}
        if input_type is not None:
            kwargs["input_type"] = input_type
        if truncation is not None:
            kwargs["truncation"] = truncation
        if dimensions is not None:
            kwargs["output_dimension"] = dimensions

        try:
            response = await self.client.embed(list(values), model=model, **kwargs)
        except voyageai.error.VoyageError as ex:
            self._raise_sdk_error(ex)
        except IntegrationException:
            raise
        except Exception as ex:
            raise IntegrationException(f"Voyage AI embeddings request failed: {ex}", inner_exception=ex) from ex

        try:
            vectors = response.embeddings
            if len(vectors) != len(values):
                raise ValueError(f"expected {len(values)} embeddings, received {len(vectors)}")
            embeddings: list[Embedding[list[float]]] = []
            for raw_vector in vectors:
                vector = [float(value) for value in raw_vector]
                if not vector or (dimensions is not None and len(vector) != dimensions):
                    raise ValueError(f"unexpected embedding length {len(vector)}")
                embeddings.append(Embedding(vector=vector, dimensions=len(vector), model=model))

            # Billing fails closed: a response without a token count is an error, never zero tokens.
            total_tokens: Any = getattr(response, "total_tokens", None)
            if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0:
                raise ValueError("response did not report a valid token count")
            return embeddings, {"input_token_count": total_tokens, "total_token_count": total_tokens}
        except Exception as ex:
            raise IntegrationInvalidResponseException(
                f"Voyage AI embeddings response was invalid: {ex}",
                inner_exception=ex,
            ) from ex

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: VoyageAIEmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[list[float], VoyageAIEmbeddingOptionsT]:
        """Call the Voyage AI embeddings API.

        Inputs beyond ``MAX_BATCH_SIZE`` are sent as consecutive requests. Vectors keep the order of
        ``values``, usage is summed, and a failed request fails the whole call. Per-request token
        ceilings are enforced by the service, which rejects an over-limit request with ``400``.

        Args:
            values: The text values to generate embeddings for.
            options: Optional embedding generation options.

        Returns:
            Generated embeddings with usage metadata.

        Raises:
            ValueError: If the model is missing or unsupported, ``dimensions`` is not supported by
                the model, or ``input_type`` is not ``"document"`` or ``"query"``.
            IntegrationInvalidAuthException: If Voyage AI rejects the credentials.
            IntegrationInvalidRequestException: If Voyage AI rejects the request.
            IntegrationInvalidResponseException: If the response is malformed, or reports no token count.
            IntegrationException: If the request fails for another reason, including rate limiting.
        """
        if not values:
            return GeneratedEmbeddings([], options=options)

        opts: dict[str, Any] = options or {}  # type: ignore
        model = opts.get("model") or self.model
        dimensions = opts.get("dimensions")
        input_type = opts.get("input_type")
        truncation = opts.get("truncation")
        _validate_model_and_dimensions(model, dimensions)
        if input_type is not None and input_type not in _INPUT_TYPES:
            raise ValueError(f"Unsupported Voyage AI input_type {input_type!r}; use 'document' or 'query'.")
        if truncation is not None and not isinstance(truncation, bool):
            raise ValueError(f"truncation must be a bool, not {truncation!r}.")

        mark_feature_used(FeatureIndex.VOYAGEAI)
        embeddings: list[Embedding[list[float]]] = []
        usage: UsageDetails | None = None
        for batch in _batches(values):
            batch_embeddings, batch_usage = await self._embed_batch(
                batch, model=model, input_type=input_type, truncation=truncation, dimensions=dimensions
            )
            embeddings.extend(batch_embeddings)
            usage = add_usage_details(usage, batch_usage)
        return GeneratedEmbeddings(embeddings, options=options, usage=usage)


class VoyageAIEmbeddingClient(
    EmbeddingTelemetryLayer[str, list[float], VoyageAIEmbeddingOptionsT],
    RawVoyageAIEmbeddingClient[VoyageAIEmbeddingOptionsT],
    Generic[VoyageAIEmbeddingOptionsT],
):
    """Voyage AI embedding client with telemetry support.

    Keyword Args:
        model: The Voyage AI embedding model (for example ``"voyage-4"``). Falls back to
            ``VOYAGE_EMBEDDING_MODEL``. There is no default model.
        api_key: The Voyage AI API key. Falls back to ``VOYAGE_API_KEY``.
        base_url: Endpoint override. Falls back to ``VOYAGE_BASE_URL``.
        client: Optional pre-configured ``AsyncClient``.
        additional_properties: Additional properties stored on the client instance.
        otel_provider_name: Optional telemetry provider name override.
        env_file_path: Path to a ``.env`` file for settings.
        env_file_encoding: Encoding of that ``.env`` file.

    Examples:
        .. code-block:: python

            from agent_framework_voyageai import VoyageAIEmbeddingClient

            # Set VOYAGE_API_KEY (and optionally VOYAGE_EMBEDDING_MODEL) in the environment.
            client = VoyageAIEmbeddingClient(model="voyage-4")

            result = await client.get_embeddings(
                ["Hello, world!"],
                options={"input_type": "document", "dimensions": 512},
            )
            print(result[0].vector)
    """

    OTEL_PROVIDER_NAME: ClassVar[str] = "voyageai"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretString | None = None,
        base_url: str | None = None,
        client: AsyncClient | None = None,
        additional_properties: dict[str, Any] | None = None,
        otel_provider_name: str | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a Voyage AI embedding client with telemetry."""
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            client=client,
            additional_properties=additional_properties,
            otel_provider_name=otel_provider_name,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
