# Voyage AI samples

`voyage_embeddings.py` generates single and batched embeddings, then indexes two notes in an
in-memory collection and retrieves the closer one. Stored notes are embedded with
`input_type="document"` and the search text with `input_type="query"`.

Set `VOYAGE_API_KEY` to your Voyage AI API key. See the [package README](../README.md) for the
other settings and the supported models.
