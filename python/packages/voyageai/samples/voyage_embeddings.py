# Copyright (c) Microsoft. All rights reserved.

import asyncio
from dataclasses import dataclass
from typing import Annotated

from agent_framework import InMemoryCollection, VectorStoreField, vectorstoremodel

from agent_framework_voyageai import VoyageAIEmbeddingClient

"""Generate embeddings with Voyage AI, then retrieve with them.

Set ``VOYAGE_API_KEY`` to your Voyage AI API key, then run from python/:
    uv run --package agent-framework-voyageai python packages/voyageai/samples/voyage_embeddings.py

The model is always named explicitly; there is no default.
"""


@vectorstoremodel(collection_name="voyage-notes")
@dataclass
class Note:
    id: Annotated[str, VectorStoreField("key")]
    text: Annotated[str, VectorStoreField("data")]
    vector: Annotated[str | list[float] | None, VectorStoreField("vector", dimensions=1024)] = None


async def main() -> None:
    """Generate embeddings with Voyage AI."""
    # The API key is read from VOYAGE_API_KEY.
    client = VoyageAIEmbeddingClient(model="voyage-4")

    # 1. Generate a single embedding.
    result = await client.get_embeddings(["Hello, world!"])
    print(f"Single embedding dimensions: {result[0].dimensions}")
    print(f"Model: {result[0].model}")
    print(f"Usage: {result.usage}")
    print()

    # 2. Generate smaller embeddings for several inputs (Matryoshka dimensions).
    texts = [
        "The weather is sunny today.",
        "It is raining outside.",
        "Machine learning is fascinating.",
    ]
    result = await client.get_embeddings(texts, options={"dimensions": 256})
    print(f"Batch of {len(result)} embeddings, each with {result[0].dimensions} dimensions")
    print()

    # 3. Retrieval: embed stored documents as documents and search text as a query.
    #    The vector store forwards input_type to each embedding request.
    collection: InMemoryCollection[str, Note] = InMemoryCollection(Note, embedding_generator=client)
    await collection.ensure_collection_exists()
    sources = {
        "one": "Voyage AI provides embedding and reranking models for retrieval.",
        "two": "The weather is sunny today.",
    }
    # The vector field holds the text to embed until the collection replaces it with the vector.
    notes = [Note(note_id, text, text) for note_id, text in sources.items()]
    await collection.upsert(notes, embeddings_options={"input_type": "document"})
    results = await collection.search(
        "Which models help with retrieval?", embeddings_options={"input_type": "query"}, top=1
    )
    async for item in results:
        print(f"Best match: {item['record'].text}")


if __name__ == "__main__":
    asyncio.run(main())
