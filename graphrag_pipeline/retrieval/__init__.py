"""graphrag_pipeline.retrieval — natural-language retrieval layer.

Four-layer architecture:
  Layer 0 — classifier.py     : query intent classification
  Layer 1 — entity_gateway.py : entity resolution
  Layer 2A — query_builder.py : analytical Cypher templates
  Layer 2B — context_assembler.py : provenance context assembly
  Layer 3  — synthesis.py     : Anthropic API synthesis
  Layer 4  — web/app.py       : FastAPI endpoints
"""
