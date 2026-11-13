# Voyage AI Package (agent-framework-voyageai)

Integration with Voyage AI for text embedding generation.

## Implementation Notes

- Uses the official `voyageai` SDK (`voyageai.AsyncClient.embed`) for embedding requests. Framework
  option, usage, and exception translation stays in this package.
- There is no default model. `model` (or `VOYAGE_EMBEDDING_MODEL`, or `options["model"]`) must name a
  model in `_MODEL_DIMENSIONS`; an unknown name raises `ValueError` before any request. The table also
  keeps the SDK's local open-weight models from being selected, since those would run in-process.
- `dimensions` is validated against the model's supported sizes and sent as `output_dimension` only when
  set. `input_type` (`"document"`/`"query"`) and `truncation` are sent only when set.
- A request holds at most 1000 inputs (`MAX_BATCH_SIZE`); `get_embeddings` splits larger calls into
  consecutive requests and sums usage. Token ceilings per request are enforced by the service.
- Usage is read from the response's `usage.total_tokens`. A missing or invalid count raises
  `IntegrationInvalidResponseException`; it is never reported as zero.
- Voyage AI error bodies are `{"detail": "<text>"}`. `_error_detail` reads that field and
  `_raise_sdk_error` maps `401`/`403` to `IntegrationInvalidAuthException`, other `4xx` except `429` to
  `IntegrationInvalidRequestException`, and everything else (including `429` and `5xx`) to
  `IntegrationException`.
- The SDK retries `429`/`503`/timeouts only when `max_retries` is set. The client keeps the SDK default (no
  retries); pass a pre-configured `voyageai.AsyncClient` to change that.

## Main Classes

- **`VoyageAIEmbeddingClient`** - Embedding client for Voyage AI text models, with telemetry
- **`RawVoyageAIEmbeddingClient`** - The same client without the telemetry layer
- **`VoyageAIEmbeddingOptions`** - Options TypedDict (`model`, `dimensions`, `input_type`, `truncation`)
- **`VoyageAIEmbeddingSettings`** - TypedDict settings for Voyage AI configuration

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
