"""MCP server that provides web search via Gemini Grounding with Google Search."""

import logging
import os

from cryptography.fernet import Fernet
from fastmcp import FastMCP
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from fastmcp.server.auth.providers.jwt import JWTVerifier
from google import genai
from google.genai import types
from key_value.aio.stores.postgresql import PostgreSQLStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
MCP_BASE_URL = os.getenv("MCP_BASE_URL", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_CLOUD_PROJECT:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")

_gemini = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)


# ── Auth ─────────────────────────────────────────────────────────────────────


def _build_storage(jwt_signing_key: str) -> FernetEncryptionWrapper | None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    if not jwt_signing_key:
        raise RuntimeError("MCP_JWT_SIGNING_KEY must be set when DATABASE_URL is configured")
    encryption_key = derive_jwt_key(
        high_entropy_material=jwt_signing_key,
        salt="fastmcp-storage-encryption-key",
    )
    return FernetEncryptionWrapper(
        store=PostgreSQLStore(url=database_url),
        fernet=Fernet(key=encryption_key),
    )


def _require_env(name: str) -> str:
    value = os.getenv(name, "")
    if not value:
        raise RuntimeError(f"{name} must be set when COGNITO_USER_POOL_ID is configured")
    return value


def _build_cognito_auth() -> OAuthProxy | None:
    pool_id = os.getenv("COGNITO_USER_POOL_ID")
    if not pool_id:
        log.info("No COGNITO_USER_POOL_ID set — running without auth")
        return None

    region = _require_env("COGNITO_REGION")
    client_id = _require_env("COGNITO_CLIENT_ID")
    client_secret = _require_env("COGNITO_CLIENT_SECRET")
    domain = _require_env("COGNITO_DOMAIN")

    cognito_base_url = f"https://{domain}.auth.{region}.amazoncognito.com"
    issuer_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"

    scopes_raw = os.getenv("COGNITO_SCOPES", "")
    required_scopes = [s.strip() for s in scopes_raw.split() if s.strip()] or None

    # Stable random string for signing JWTs. Generate with: openssl rand -base64 32
    jwt_signing_key = os.getenv("MCP_JWT_SIGNING_KEY")
    auth = OAuthProxy(
        upstream_authorization_endpoint=f"{cognito_base_url}/oauth2/authorize",
        upstream_token_endpoint=f"{cognito_base_url}/oauth2/token",
        upstream_client_id=client_id,
        upstream_client_secret=client_secret,
        token_verifier=JWTVerifier(
            jwks_uri=f"{issuer_url}/.well-known/jwks.json",
            issuer=issuer_url,
            required_scopes=required_scopes,
        ),
        base_url=MCP_BASE_URL,
        token_endpoint_auth_method="client_secret_basic",
        forward_pkce=True,
        jwt_signing_key=jwt_signing_key,
        client_storage=_build_storage(jwt_signing_key or ""),
    )

    log.info("Cognito OAuth proxy enabled: pool=%s region=%s domain=%s", pool_id, region, domain)
    return auth


mcp = FastMCP("gemini-websearch", auth=_build_cognito_auth())


# ── Response formatting ──────────────────────────────────────────────────────
#
# Gemini Grounding responses contain the answer text plus metadata linking
# spans of text back to web sources. We format this as markdown so MCP
# clients (LLMs) can pass it directly to the user:
#
#   Python 3.14 was released in October 2025. [1](https://python.org) [2](https://en.wikipedia.org)
#
#   Sources:
#   - [Python.org](https://python.org/downloads/)
#   - [Wikipedia](https://en.wikipedia.org/wiki/Python)


def _format_response(response) -> str:
    text = response.text or ""

    if not response.candidates:
        return text

    metadata = response.candidates[0].grounding_metadata
    if not metadata:
        return text

    chunks = metadata.grounding_chunks or []
    supports = metadata.grounding_supports or []

    # Insert inline citation links into the text at each supported span.
    if supports and chunks:
        text = _build_cited_text(text, supports, chunks)

    # Append a sources list at the end.
    sources = [
        f"- [{chunk.web.title}]({chunk.web.uri})"
        for chunk in chunks
        if chunk.web
    ]
    if sources:
        text += "\n\nSources:\n" + "\n".join(sources)

    return text


def _build_cited_text(text: str, supports, chunks) -> str:
    # Each "support" marks a text span (start_index..end_index) and the chunk
    # indices that back it. We insert "[N](url)" links after each span.
    # Process from end to start so earlier offsets stay valid after insertion.
    cited = text
    for support in sorted(supports, key=lambda s: s.segment.end_index, reverse=True):
        if not support.grounding_chunk_indices:
            continue
        links = [
            f"[{i + 1}]({chunks[i].web.uri})"
            for i in support.grounding_chunk_indices
            if i < len(chunks) and chunks[i].web
        ]
        if links:
            end = support.segment.end_index
            cited = cited[:end] + " " + ", ".join(links) + cited[end:]
    return cited


# ── Tools ────────────────────────────────────────────────────────────────────

_GOOGLE_SEARCH_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())],
)


@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using Gemini Grounding with Google Search.

    Returns up-to-date information with citations from real web sources.
    Use this for any question that benefits from current, factual information.

    Args:
        query: The search query or question to answer.
    """
    response = await _gemini.aio.models.generate_content(
        model=GEMINI_MODEL, contents=query, config=_GOOGLE_SEARCH_CONFIG
    )
    return _format_response(response)


# ── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Gemini Web Search MCP server (streamable-http)")
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
