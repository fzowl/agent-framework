# Get Started with Microsoft Agent Framework Voyage AI

An alpha [Voyage AI](https://www.voyageai.com/) text embedding integration for Microsoft Agent Framework. Please install this package:

```bash
pip install agent-framework-voyageai --pre
```

and see the [README](https://github.com/microsoft/agent-framework/tree/main/python/README.md) for more information.

See the [Voyage AI samples](samples/README.md) for a runnable example.

## Embedding Client

The `VoyageAIEmbeddingClient` generates text embeddings, and `VoyageAIContextualizedEmbeddingClient` generates
contextualized chunk embeddings, with the official
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

A request holds at most 1000 inputs, and each model has a documented ceiling on the tokens of one request
(1M for `voyage-4-lite` and `voyage-3.5-lite`, 320K for `voyage-4` and `voyage-3.5`, 120K for the others;
none is documented for `voyage-code-4`, so only the input count is limited there). The client splits larger calls
into consecutive requests within both limits, returns the vectors in order, and sums usage.

Tokens are counted with the SDK's `count_tokens`, which uses the model's own tokenizer. The tokenizer is
downloaded from the Hugging Face Hub on first use. If tokens cannot be counted, the call raises
`IntegrationException` instead of sending an unsplit request. `voyage-code-4` has no documented ceiling, so
calls to it are split by input count only and never load a tokenizer.

Usage comes from the token count the service reports. A response without one raises
`IntegrationInvalidResponseException` instead of reporting zero tokens.

Only the models in the table above are accepted. The `voyageai` SDK can also run some open-weight models
on the local machine, and the client does not select them.

## Contextualized Embedding Client

Contextualized embeddings give each chunk of a document a vector that also reflects the rest of the document.
One document yields several vectors, which does not fit `get_embeddings` (one vector per input), so
`VoyageAIContextualizedEmbeddingClient` has its own method and result type. It has no telemetry layer.

```python
from agent_framework_voyageai import VoyageAIContextualizedEmbeddingClient

# The model is always named explicitly: pass it here, or set VOYAGE_CONTEXTUALIZED_EMBEDDING_MODEL.
client = VoyageAIContextualizedEmbeddingClient(model="voyage-context-4")

documents = await client.get_contextualized_embeddings(
    ["A long document ...", "Another long document ..."],
    options={"input_type": "document", "dimensions": 512},
)
for document in documents:
    for chunk, embedding in zip(document.chunks or [], document.embeddings):
        print(chunk[:40], embedding.dimensions)

queries = await client.get_contextualized_embeddings(["Why do cats purr?"], options={"input_type": "query"})
```

| Model | `dimensions` |
|---|---|
| `voyage-context-4`, `voyage-context-3` | 256, 512, 1024 (default), 2048 |

- `input_type` is required. `"document"` sends each value as one whole document (a flat list of strings, one
  string per document) with `enable_auto_chunking=True` and `chunk_size=32_000`, so Voyage AI does the
  chunking. A pre-chunked document, such as a list of strings as one value, raises `ValueError`: the service
  rejects it when it chunks itself. `"query"` sends each value as one query and does not chunk.
- Each result holds one vector per chunk in `embeddings`, and the text of each chunk in `chunks` when the service
  returned it. Queries have one vector and `chunks` is `None`.
- Requests are split by the 1000-input and 120K-token limits like the text client, and the same fail-closed
  usage and error mapping apply.
