# Get Started with Microsoft Agent Framework Voyage AI

An alpha [Voyage AI](https://www.voyageai.com/) text embedding integration for Microsoft Agent Framework. Please install this package:

```bash
pip install agent-framework-voyageai --pre
```

and see the [README](https://github.com/microsoft/agent-framework/tree/main/python/README.md) for more information.

See the [Voyage AI samples](samples/README.md) for a runnable example.

## Embedding Client

The `VoyageAIEmbeddingClient` generates text embeddings with the official
[`voyageai`](https://pypi.org/project/voyageai/) SDK.

### Quick Start

```python
from agent_framework_voyageai import VoyageAIEmbeddingClient

# Reads VOYAGE_API_KEY from the environment. The model is always named explicitly:
# pass it here, or set VOYAGE_EMBEDDING_MODEL.
client = VoyageAIEmbeddingClient(model="voyage-4")

result = await client.get_embeddings(
    ["Hello, world!", "How are you?"],
    options={"input_type": "document"},
)
for embedding in result:
    print(f"Dimensions: {embedding.dimensions}")
print(result.usage)
```

### Configuration

| Environment Variable | Description |
|---|---|
| `VOYAGE_API_KEY` | Your Voyage AI API key |
| `VOYAGE_EMBEDDING_MODEL` | Embedding model name (for example `voyage-4`). There is no default model |
| `VOYAGE_BASE_URL` | Optional endpoint override |

### Models

There is no default model. The client rejects a missing model, or a name outside this table, before
it sends a request.

| Model | `dimensions` |
|---|---|
| `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-code-4` | 256, 512, 1024 (default), 2048 |
| `voyage-3-large`, `voyage-3.5`, `voyage-3.5-lite`, `voyage-code-3` | 256, 512, 1024 (default), 2048 |
| `voyage-finance-2`, `voyage-law-2` | 1024 |

### Options

`VoyageAIEmbeddingOptions` adds two Voyage AI fields to the common embedding options:

| Option | Values | Purpose |
|---|---|---|
| `input_type` | `"document"`, `"query"` | Voyage AI prepends a retrieval prompt for each. Omitted, nothing is prepended |
| `truncation` | `True`, `False` | Shorten over-length input, or reject it. Omitted, the service default (`True`) applies |

`dimensions` is sent as `output_dimension`, and only when set. For retrieval, embed stored documents with
`input_type="document"` and search text with `input_type="query"`. Vector collections forward
`embeddings_options` to each request:

```python
await collection.upsert(notes, embeddings_options={"input_type": "document"})
results = await collection.search("How do I call a model?", embeddings_options={"input_type": "query"})
```

### Requests and usage

Voyage AI accepts at most 1000 inputs per request, so the client sends larger calls as consecutive requests,
returns the vectors in order, and sums usage. The service also caps the tokens in one request per model and
answers an over-limit request with `400`, which surfaces as `IntegrationInvalidRequestException`.

Usage comes from the token count the service reports. A response without one raises
`IntegrationInvalidResponseException` instead of reporting zero tokens.

Only the models in the table above are accepted. The `voyageai` SDK can also run some open-weight models
on the local machine, and the client does not select them.
