# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypedDict

import voyageai.error
from agent_framework import Embedding, UsageDetails, add_usage_details, load_settings
from agent_framework._settings import SecretString
from agent_framework._telemetry import mark_feature_used
from agent_framework.exceptions import IntegrationException
from voyageai.client_async import AsyncClient

from ._feature_usage import FeatureIndex
from ._shared import (
    INPUT_TYPES,
    MATRYOSHKA_DIMENSIONS,
    VOYAGEAI_API_BASE_URL,
    create_sdk_client,
    invalid_response,
    raise_sdk_error,
    split_into_requests,
    validate_dimensions,
    validate_input_type,
)

__all__ = [
    "VoyageAIContextualizedEmbedding",
    "VoyageAIContextualizedEmbeddingClient",
    "VoyageAIContextualizedEmbeddingOptions",
    "VoyageAIContextualizedEmbeddingSettings",
    "VoyageAIContextualizedEmbeddings",
]

_AUTO_CHUNK_SIZE: Final = 32_000
"""Target size, in tokens, of the chunks Voyage AI cuts a document into. 32,000 is the service maximum."""

_MAX_REQUEST_TOKENS: Final = 120_000
"""The total tokens Voyage AI accepts across the inputs of one contextualized embeddings request."""

_MODELS: Final[Mapping[str, tuple[int, ...]]] = {
    "voyage-context-4": MATRYOSHKA_DIMENSIONS,
    "voyage-context-3": MATRYOSHKA_DIMENSIONS,
}
"""The contextualized embedding models this client accepts, with the ``dimensions`` each one supports.

The model is always named by the caller; there is no default.
"""


class VoyageAIContextualizedEmbeddingOptions(TypedDict, total=False):
    """Options for contextualized embeddings.

    ``model`` and ``input_type`` are required, either here or, for ``model``, on the client.

    Keys:
        model: ``"voyage-context-4"`` or ``"voyage-context-3"``.
        input_type: ``"document"`` embeds each value as one whole document that Voyage AI splits into
            chunks and embeds in the context of the whole document. ``"query"`` embeds each value as one
            query and does not chunk. Voyage AI only chunks documents, so there is no default.
        dimensions: One of 256, 512, 1024 (the service default) or 2048, sent as ``output_dimension``
            and only when set.
    """

    model: str
    input_type: Literal["document", "query"]
    dimensions: int


class VoyageAIContextualizedEmbeddingSettings(TypedDict, total=False):
    """Voyage AI contextualized embedding settings.

    Settings are resolved in this order: explicit keyword arguments, values from an
    explicitly provided .env file, then environment variables with the prefix 'VOYAGE_'.

    Keys:
        api_key: The Voyage AI API key. (Env var VOYAGE_API_KEY)
        contextualized_embedding_model: The model, ``voyage-context-4`` or ``voyage-context-3``. There is
            no default. (Env var VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL)
        base_url: Optional endpoint override. (Env var VOYAGE_BASE_URL)
    """

    api_key: SecretString | None
    contextualized_embedding_model: str | None
    base_url: str | None


@dataclass
class VoyageAIContextualizedEmbedding:
    """The embeddings of one input value.

    Attributes:
        embeddings: One vector per chunk, in document order. A query has exactly one.
        chunks: The text of each chunk, in the same order, as Voyage AI split it. ``None`` when the
            service did not return chunk text, which is always the case for ``input_type="query"``.
    """

    embeddings: list[Embedding[list[float]]]
    chunks: list[str] | None = None


class VoyageAIContextualizedEmbeddings(list[VoyageAIContextualizedEmbedding]):
    """The contextualized embeddings of a call, one entry per input value and in input order.

    Attributes:
        options: The options the call was made with.
        usage: Token usage summed over every request of the call.
    """

    def __init__(
        self,
        embeddings: Sequence[VoyageAIContextualizedEmbedding] | None = None,
        *,
        options: VoyageAIContextualizedEmbeddingOptions | None = None,
        usage: UsageDetails | None = None,
    ) -> None:
        super().__init__(embeddings or [])
        self.options = options
        self.usage = usage


def _model_dimensions(model: str, dimensions: Any) -> None:
    supported = _MODELS.get(model)
    if supported is None:
        raise ValueError(
            f"Unsupported Voyage AI contextualized embedding model {model!r}. Supported models: {', '.join(_MODELS)}."
        )
    validate_dimensions(model, dimensions, supported)


class VoyageAIContextualizedEmbeddingClient:
    """Voyage AI contextualized embedding client.

    Contextualized embeddings give every chunk of a document a vector that also reflects the rest of
    the document. A document yields several vectors, so this does not fit ``BaseEmbeddingClient``,
    which returns one vector per input. It has its own method and result type, and no telemetry layer.

    Documents are always sent as one string each, with ``enable_auto_chunking`` on and a chunk size of
    32,000 tokens, and Voyage AI returns the chunk text with the vectors. Queries are sent unchunked.

    Keyword Args:
        model: ``"voyage-context-4"`` or ``"voyage-context-3"``. Falls back to
            ``VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL``. There is no default.
        api_key: The Voyage AI API key. Falls back to ``VOYAGE_API_KEY``.
        base_url: Endpoint override. Falls back to ``VOYAGE_BASE_URL``. Ignored when ``client`` is provided.
        client: Optional pre-configured ``voyageai.AsyncClient``. It owns its credentials, so ``api_key``
            is not required.
        env_file_path: Path to a ``.env`` file for settings.
        env_file_encoding: Encoding of that ``.env`` file.

    Raises:
        SettingNotFoundError: If no model is supplied.
        ValueError: If the model is not a supported contextualized model, or no API key is available and
            no ``client`` was passed.

    Examples:
        .. code-block:: python

            from agent_framework_voyageai import VoyageAIContextualizedEmbeddingClient

            client = VoyageAIContextualizedEmbeddingClient(model="voyage-context-4")

            documents = await client.get_contextualized_embeddings(
                ["A long document ...", "Another long document ..."],
                options={"input_type": "document"},
            )
            for chunk, vector in zip(documents[0].chunks or [], documents[0].embeddings):
                print(chunk[:40], len(vector.vector))
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretString | None = None,
        base_url: str | None = None,
        client: AsyncClient | None = None,
        env_file_path: str | None = None,
        env_file_encoding: str | None = None,
    ) -> None:
        """Initialize a Voyage AI contextualized embedding client."""
        if client is not None and not isinstance(client, AsyncClient):
            raise TypeError(f"The 'client' parameter accepts a voyageai AsyncClient; got {type(client).__name__}.")

        settings = load_settings(
            VoyageAIContextualizedEmbeddingSettings,
            env_prefix="VOYAGE_",
            required_fields=["contextualized_embedding_model"],
            contextualized_embedding_model=model,
            api_key=api_key,
            base_url=base_url,
            env_file_path=env_file_path,
            env_file_encoding=env_file_encoding,
        )
        resolved_model: str = settings["contextualized_embedding_model"]  # type: ignore[assignment]
        _model_dimensions(resolved_model, None)
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

    def service_url(self) -> str:
        """Get the URL of the service."""
        return self.base_url or VOYAGEAI_API_BASE_URL

    async def _embed_batch(
        self,
        values: Sequence[str],
        *,
        model: str,
        input_type: str,
        dimensions: int | None,
    ) -> tuple[list[VoyageAIContextualizedEmbedding], UsageDetails]:
        kwargs: dict[str, Any] = {}
        if dimensions is not None:
            kwargs["output_dimension"] = dimensions
        if input_type == "document":
            # One string per document: the service rejects pre-chunked lists when it chunks itself.
            inputs: list[Any] = list(values)
            kwargs["enable_auto_chunking"] = True
            kwargs["chunk_size"] = _AUTO_CHUNK_SIZE
        else:
            inputs = [[value] for value in values]

        try:
            response = await self.client.contextualized_embed(inputs, model=model, input_type=input_type, **kwargs)
        except voyageai.error.VoyageError as ex:
            raise_sdk_error(ex, "contextualized embeddings")
        except (AttributeError, KeyError, TypeError) as ex:
            # The SDK reads the usage and vectors out of the body while building its result.
            raise invalid_response(ex, "contextualized embeddings") from ex
        except Exception as ex:
            raise IntegrationException(
                f"Voyage AI contextualized embeddings request failed: {ex}", inner_exception=ex
            ) from ex

        try:
            if len(response.results) != len(values):
                raise ValueError(f"expected {len(values)} results, received {len(response.results)}")
            results: list[VoyageAIContextualizedEmbedding] = []
            for result in response.results:
                embeddings: list[Embedding[list[float]]] = []
                for raw_vector in result.embeddings:
                    vector = [float(value) for value in raw_vector]
                    if not vector or (dimensions is not None and len(vector) != dimensions):
                        raise ValueError(f"unexpected embedding length {len(vector)}")
                    embeddings.append(Embedding(vector=vector, dimensions=len(vector), model=model))
                if not embeddings:
                    raise ValueError("a result holds no embeddings")
                chunks = list(result.chunk_texts) if result.chunk_texts is not None else None
                if chunks is not None and len(chunks) != len(embeddings):
                    raise ValueError(f"{len(embeddings)} embeddings but {len(chunks)} chunks")
                results.append(VoyageAIContextualizedEmbedding(embeddings=embeddings, chunks=chunks))

            total_tokens = response.total_tokens
            return results, {"input_token_count": total_tokens, "total_token_count": total_tokens}
        except Exception as ex:
            raise invalid_response(ex, "contextualized embeddings") from ex

    async def get_contextualized_embeddings(
        self,
        values: Sequence[str],
        *,
        options: VoyageAIContextualizedEmbeddingOptions,
    ) -> VoyageAIContextualizedEmbeddings:
        """Call the Voyage AI contextualized embeddings API.

        Args:
            values: With ``input_type="document"``, whole documents, one string each; Voyage AI chunks
                them. With ``input_type="query"``, one query per string.
            options: The call options. ``input_type`` is required; ``model`` is required unless the
                client was created with one.

        Returns:
            One entry per value, in order, with a vector per chunk, the chunk text when Voyage AI
            returned it, and the token usage summed over the requests of the call.

        Raises:
            ValueError: If the model or ``input_type`` is missing or unsupported, ``dimensions`` is not
                supported, or a value is not a string (a document must not be pre-chunked).
            IntegrationInvalidAuthException: If Voyage AI rejects the credentials.
            IntegrationInvalidRequestException: If Voyage AI rejects the request.
            IntegrationInvalidResponseException: If the response is malformed, or reports no token count.
            IntegrationException: If the request fails for another reason, including rate limiting, or
                if the tokenizer needed to split a large call cannot be loaded.
        """
        opts: dict[str, Any] = options  # type: ignore[assignment]
        model = opts.get("model") or self.model
        dimensions = opts.get("dimensions")
        input_type = opts.get("input_type")
        _model_dimensions(model, dimensions)
        if input_type is None:
            raise ValueError(
                f"input_type is required for contextualized embeddings; use one of {list(INPUT_TYPES)}. "
                "Voyage AI chunks documents only with input_type='document'."
            )
        validate_input_type(input_type)
        if isinstance(values, str):
            raise ValueError("values must be a sequence of strings, not a single string; wrap one document in a list.")
        for index, value in enumerate(values):
            if not isinstance(value, str):
                raise ValueError(
                    f"values[{index}] is a {type(value).__name__}; send each document as one string, "
                    "Voyage AI does the chunking."
                )
        if not values:
            return VoyageAIContextualizedEmbeddings(options=options)

        mark_feature_used(FeatureIndex.VOYAGEAI)
        results: list[VoyageAIContextualizedEmbedding] = []
        usage: UsageDetails | None = None
        requests = await split_into_requests(self.client, model, values, max_request_tokens=_MAX_REQUEST_TOKENS)
        for batch in requests:
            batch_results, batch_usage = await self._embed_batch(
                batch, model=model, input_type=input_type, dimensions=dimensions
            )
            results.extend(batch_results)
            usage = add_usage_details(usage, batch_usage)
        return VoyageAIContextualizedEmbeddings(results, options=options, usage=usage)
