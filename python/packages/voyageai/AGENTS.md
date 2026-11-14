# Voyage AI Package (agent-framework-voyageai)

Integration with Voyage AI for text embedding generation and contextualized chunk embeddings.

## Implementation Notes

- Uses the official `voyageai` SDK (`voyageai.AsyncClient.embed`) for embedding requests. Framework
  option, usage, and exception translation stays in this package.
- There is no default model. `model` (or `VOYAGE_EMBEDDING_MODEL`, or `options["model"]`) must name a
  model in `_MODELS`; an unknown name raises `ValueError` before any request. The table also
  keeps the SDK's local open-weight models from being selected, since those would run in-process.
- `dimensions` is validated against the model's supported sizes and sent as `output_dimension` only when
  set. `input_type` (`"document"`/`"query"`) and `truncation` are sent only when set.
- A request holds at most 1000 inputs (`MAX_BATCH_SIZE`) and, per model, at most the tokens in
  `_ModelSpec.max_request_tokens` (`None` for `voyage-code-4`: no ceiling is documented, so none is assumed).
  `split_into_requests` splits larger calls into consecutive requests and `get_embeddings` sums usage. Tokens
  are counted with the SDK's `client.count_tokens` (its tokenizer is fetched from the Hugging Face Hub on first
  use); there is no estimation. A counting failure raises instead of sending an unsplit request.
- Contextualized embeddings do not fit `BaseEmbeddingClient` (several vectors per input), so
  `VoyageAIContextualizedEmbeddingClient.get_contextualized_embeddings` has its own result type
  (`VoyageAIContextualizedEmbeddings`, one `VoyageAIContextualizedEmbedding` per value with `embeddings` and
  `chunks`) and no telemetry layer. `input_type` is required. Documents (`"document"`) are always a flat
  `list[str]` with `enable_auto_chunking=True` and `chunk_size=32_000`; the API rejects auto-chunking with any
  other input type or with pre-chunked inner lists. Queries are sent as `[[query], ...]` without chunking.
- Usage is read from the response's `usage.total_tokens`. A missing or invalid count raises
  `IntegrationInvalidResponseException`; it is never reported as zero.
- Voyage AI error bodies are `{"detail": "<text>"}`. `_shared.error_detail` reads that field and
  `_shared.raise_sdk_error` maps `401`/`403` to `IntegrationInvalidAuthException`, other `4xx` except `429` to
  `IntegrationInvalidRequestException`, and everything else (including `429` and `5xx`) to
  `IntegrationException`.
- The SDK retries `429`/`503`/timeouts only when `max_retries` is set. The client keeps the SDK default (no
  retries); pass a pre-configured `voyageai.AsyncClient` to change that.

## Main Classes

- **`VoyageAIEmbeddingClient`** - Embedding client for Voyage AI text models, with telemetry
- **`RawVoyageAIEmbeddingClient`** - The same client without the telemetry layer
- **`VoyageAIEmbeddingOptions`** - Options TypedDict (`model`, `dimensions`, `input_type`, `truncation`)
- **`VoyageAIEmbeddingSettings`** - TypedDict settings for Voyage AI configuration
- **`VoyageAIContextualizedEmbeddingClient`** - Contextualized chunk embeddings for `voyage-context-4`/`-3`
- **`VoyageAIContextualizedEmbeddingOptions`** / **`VoyageAIContextualizedEmbeddingSettings`** - Its options and settings
- **`VoyageAIContextualizedEmbeddings`** / **`VoyageAIContextualizedEmbedding`** - Its result types

## Usage

```python
from agent_framework_voyageai import VoyageAIEmbeddingClient

# Requires VOYAGE_API_KEY environment variable (or pass api_key= directly)
client = VoyageAIEmbeddingClient(model="voyage-4")
result = await client.get_embeddings(["Hello, world!"], options={"input_type": "document"})
print(result[0].vector)
```

## Import Path

The package is alpha, so there is no `agent_framework.voyageai` namespace in core yet.

```python
from agent_framework_voyageai import VoyageAIEmbeddingClient
```
