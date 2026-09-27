"""API-mapping gateway.

Clients speak a public API (OpenAI Chat Completions, Anthropic Messages); each
request is parsed into the canonical form in ``ir``, routed to whichever
inference target is active, and sent through an engine adapter. Adding an API
or an engine means one adapter against ``ir``, not one per API/engine pair.
"""
