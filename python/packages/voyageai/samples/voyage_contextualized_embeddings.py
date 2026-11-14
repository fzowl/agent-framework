# Copyright (c) Microsoft. All rights reserved.

import asyncio

from agent_framework_voyageai import VoyageAIContextualizedEmbeddingClient

"""Generate contextualized chunk embeddings with Voyage AI.

Set ``VOYAGE_API_KEY`` to your Voyage AI API key, then run from python/:
    uv run --package agent-framework-voyageai python packages/voyageai/samples/voyage_contextualized_embeddings.py

The model is always named explicitly; there is no default. Documents are sent as one string each and
Voyage AI chunks them, so the result also carries the text of every chunk.
"""


async def main() -> None:
    """Generate contextualized embeddings with Voyage AI."""
    # The API key is read from VOYAGE_API_KEY.
    client = VoyageAIContextualizedEmbeddingClient(model="voyage-context-4")

    # 1. Embed whole documents. Each one yields a vector per chunk.
    documents = [
        "Cats purr when they are content. Some cats also purr when they are hurt or anxious.",
        "Dogs wag their tails when they are excited. A slow wag can also signal uncertainty.",
    ]
    result = await client.get_contextualized_embeddings(
        documents, options={"input_type": "document", "dimensions": 512}
    )
    for document in result:
        for chunk, embedding in zip(document.chunks or [], document.embeddings, strict=True):
            print(f"{embedding.dimensions} dimensions: {chunk}")
    print(f"Usage: {result.usage}")
    print()

    # 2. Embed a query. Queries are not chunked, so there is one vector and no chunk text.
    queries = await client.get_contextualized_embeddings(
        ["Why do cats purr?"], options={"input_type": "query", "dimensions": 512}
    )
    print(f"Query vectors: {len(queries[0].embeddings)}, chunks: {queries[0].chunks}")


if __name__ == "__main__":
    asyncio.run(main())
