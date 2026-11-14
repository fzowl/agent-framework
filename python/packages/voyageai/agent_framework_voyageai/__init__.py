# Copyright (c) Microsoft. All rights reserved.

import importlib.metadata

from ._contextualized_embedding_client import (
    VoyageAIContextualizedEmbedding,
    VoyageAIContextualizedEmbeddingClient,
    VoyageAIContextualizedEmbeddingOptions,
    VoyageAIContextualizedEmbeddings,
    VoyageAIContextualizedEmbeddingSettings,
)
from ._embedding_client import (
    RawVoyageAIEmbeddingClient,
    VoyageAIEmbeddingClient,
    VoyageAIEmbeddingOptions,
    VoyageAIEmbeddingSettings,
)

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode

__all__ = [
    "RawVoyageAIEmbeddingClient",
    "VoyageAIContextualizedEmbedding",
    "VoyageAIContextualizedEmbeddingClient",
    "VoyageAIContextualizedEmbeddingOptions",
    "VoyageAIContextualizedEmbeddingSettings",
    "VoyageAIContextualizedEmbeddings",
    "VoyageAIEmbeddingClient",
    "VoyageAIEmbeddingOptions",
    "VoyageAIEmbeddingSettings",
    "__version__",
]
