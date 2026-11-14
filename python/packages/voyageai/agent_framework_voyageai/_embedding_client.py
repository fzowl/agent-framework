# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Generic, Literal, TypedDict

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
from agent_framework.exceptions import IntegrationException
from agent_framework.observability import EmbeddingTelemetryLayer
from voyageai.client_async import AsyncClient

from ._feature_usage import FeatureIndex
from ._shared import (
    MATRYOSHKA_DIMENSIONS,
    VOYAGEAI_API_BASE_URL,
    create_sdk_client,
    invalid_response,
    raise_sdk_error,
    split_into_requests,
    validate_dimensions,
    validate_input_type,
)

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


@dataclass(frozen=True)
class _ModelSpec:
    dimensions: tuple[int, ...]
    max_request_tokens: int | None
    """The documented ceiling on the tokens of one request, or ``None`` when Voyage AI documents none."""


_FIXED_DIMENSIONS: Final = (1024,)

_MODELS: Final[Mapping[str, _ModelSpec]] = {
    "voyage-4-large": _ModelSpec(MATRYOSHKA_DIMENSIONS, 120_000),
    "voyage-4": _ModelSpec(MATRYOSHKA_DIMENSIONS, 320_000),
    "voyage-4-lite": _ModelSpec(MATRYOSHKA_DIMENSIONS, 1_000_000),
    # No per-request token ceiling is documented for voyage-code-4, so none is assumed.
    "voyage-code-4": _ModelSpec(MATRYOSHKA_DIMENSIONS, None),
    "voyage-3-large": _ModelSpec(MATRYOSHKA_DIMENSIONS, 120_000),
    "voyage-3.5": _ModelSpec(MATRYOSHKA_DIMENSIONS, 320_000),
    "voyage-3.5-lite": _ModelSpec(MATRYOSHKA_DIMENSIONS, 1_000_000),
    "voyage-code-3": _ModelSpec(MATRYOSHKA_DIMENSIONS, 120_000),
    "voyage-finance-2": _ModelSpec(_FIXED_DIMENSIONS, 120_000),
    "voyage-law-2": _ModelSpec(_FIXED_DIMENSIONS, 120_000),
}
"""The text embedding models this client accepts, with the ``dimensions`` each one supports and the
ceiling Voyage AI documents for the tokens of one request.

The model is always named by the caller; there is no default. A name outside this table is rejected
before any request is sent. The Voyage AI SDK also runs some open-weight models locally, and this
table keeps those from being selected by accident.
"""


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


def _model_spec(model: str, dimensions: Any) -> _ModelSpec:
    spec = _MODELS.get(model)
    if spec is None:
        raise ValueError(f"Unsupported Voyage AI embedding model {model!r}. Supported models: {', '.join(_MODELS)}.")
    validate_dimensions(model, dimensions, spec.dimensions)
    return spec


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
        _model_spec(resolved_model, None)
        self.model: str = resolved_model
        self.base_url: str | None = settings.get("base_url")

        if client is None:
            key = settings.get("api_key")
            if key is None or not key.get_secret_value().strip():
                raise ValueError(
                    "A Voyage AI API key is required. Set VOYAGE_API_KEY, pass api_key, "
                    "or supply a pre-configured client."
                )
            client = create_sdk_client(key.get_secret_value(), self.base_url)
        self.client = client

        super().__init__(additional_properties=additional_properties)

    def service_url(self) -> str:
        """Get the URL of the service."""
        return self.base_url or VOYAGEAI_API_BASE_URL

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
            raise_sdk_error(ex, "embeddings")
        except (AttributeError, KeyError, TypeError) as ex:
            # The SDK reads the usage and vectors out of the body while building its result. A body without
            # them, including one with no token count, surfaces here as a lookup or type error.
            raise invalid_response(ex, "embeddings") from ex
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

            # The SDK sums the reported token counts and fails on a response without one, so billing
            # never falls back to zero tokens.
            total_tokens = response.total_tokens
            return embeddings, {"input_token_count": total_tokens, "total_token_count": total_tokens}
        except Exception as ex:
            raise invalid_response(ex, "embeddings") from ex

    async def get_embeddings(
        self,
        values: Sequence[str],
        *,
        options: VoyageAIEmbeddingOptionsT | None = None,
    ) -> GeneratedEmbeddings[list[float], VoyageAIEmbeddingOptionsT]:
        """Call the Voyage AI embeddings API.

        Inputs beyond ``MAX_BATCH_SIZE``, or beyond the model's documented per-request token ceiling,
        are sent as consecutive requests. The tokens are counted with the SDK's ``count_tokens``, whose
        tokenizer is downloaded on first use.
        Vectors keep the order of ``values``, usage is summed, and a failed request fails the whole call.

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
            IntegrationException: If the request fails for another reason, including rate limiting, or
                if tokens cannot be counted to split the call.
        """
        if not values:
            return GeneratedEmbeddings([], options=options)

        opts: dict[str, Any] = options or {}  # type: ignore
        model = opts.get("model") or self.model
        dimensions = opts.get("dimensions")
        input_type = opts.get("input_type")
        truncation = opts.get("truncation")
        spec = _model_spec(model, dimensions)
        validate_input_type(input_type)
        if truncation is not None and not isinstance(truncation, bool):
            raise ValueError(f"truncation must be a bool, not {truncation!r}.")

        mark_feature_used(FeatureIndex.VOYAGEAI)
        embeddings: list[Embedding[list[float]]] = []
        usage: UsageDetails | None = None
        requests = await split_into_requests(self.client, model, values, max_request_tokens=spec.max_request_tokens)
        for batch in requests:
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
