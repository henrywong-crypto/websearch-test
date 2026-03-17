"""MCP server that provides web search via Gemini Grounding with Google Search."""

import json
import logging
import os
from typing import Any

from cryptography.fernet import Fernet
from fastmcp import FastMCP
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from fastmcp.server.auth.providers.jwt import JWTVerifier
from google import genai
from google.genai import types
from key_value.aio.stores.postgresql import PostgreSQLStore
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# --- Environment ---

MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
# Public base URL that MCP clients (e.g. Cursor) reach the server at.
# Must be HTTPS in production so OAuth redirect URIs are accepted.
MCP_BASE_URL = os.getenv("MCP_BASE_URL", f"http://{MCP_HOST}:{MCP_PORT}")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
if not GOOGLE_CLOUD_PROJECT:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set")
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Build the Gemini client once at startup. Credentials come from the VM's
# metadata server via Application Default Credentials (ADC) on GCP.
_gemini = genai.Client(
    vertexai=True,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
)

# --- Auth ---


def _build_storage(jwt_signing_key: str) -> FernetEncryptionWrapper | None:
    """Return an encrypted PostgreSQL token store, or None to use the default disk store."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return None
    if not jwt_signing_key:
        raise RuntimeError(
            "MCP_JWT_SIGNING_KEY must be set when DATABASE_URL is configured"
        )
    encryption_key = derive_jwt_key(
        high_entropy_material=jwt_signing_key,
        salt="fastmcp-storage-encryption-key",
    )
    # Redact credentials from the URL before logging.
    from urllib.parse import urlparse

    parsed = urlparse(database_url)
    redacted = parsed._replace(netloc=f"***@{parsed.hostname}:{parsed.port or 5432}")
    logger.info("Using PostgreSQL storage backend: %s", redacted.geturl())
    return FernetEncryptionWrapper(
        store=PostgreSQLStore(url=database_url),
        fernet=Fernet(key=encryption_key),
    )


def _build_cognito_auth() -> OAuthProxy | None:
    """Configure Cognito OAuth proxy if COGNITO_USER_POOL_ID is set, else return None."""
    pool_id = os.getenv("COGNITO_USER_POOL_ID")
    if not pool_id:
        logger.info("No COGNITO_USER_POOL_ID set — running without auth")
        return None

    region = os.getenv("COGNITO_REGION") or pool_id.split("_")[0]
    client_id = os.getenv("COGNITO_CLIENT_ID", "")
    client_secret = os.getenv("COGNITO_CLIENT_SECRET", "")
    domain = os.getenv("COGNITO_DOMAIN", "")  # domain prefix only, e.g. "myapp"

    if not client_id:
        raise RuntimeError(
            "COGNITO_CLIENT_ID must be set when COGNITO_USER_POOL_ID is configured"
        )
    if not client_secret:
        raise RuntimeError(
            "COGNITO_CLIENT_SECRET must be set when COGNITO_USER_POOL_ID is configured"
        )
    if not domain:
        raise RuntimeError("COGNITO_DOMAIN must be set (domain prefix, e.g. 'myapp')")

    cognito_base_url = f"https://{domain}.auth.{region}.amazoncognito.com"
    token_url = f"{cognito_base_url}/oauth2/token"
    issuer_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"

    # Only set required_scopes if explicitly configured. Cognito access tokens do
    # NOT carry openid/profile/email scopes (those are ID token scopes only), so
    # leaving required_scopes empty avoids a permanent 401 loop.
    scopes_raw = os.getenv("COGNITO_SCOPES", "")
    required_scopes = [s.strip() for s in scopes_raw.split() if s.strip()] or None

    token_verifier = JWTVerifier(
        jwks_uri=f"{issuer_url}/.well-known/jwks.json",
        issuer=issuer_url,
        # Do NOT set audience: Cognito access tokens use `client_id` instead of
        # the standard `aud` claim, so an audience check would always fail.
        required_scopes=required_scopes,
    )

    jwt_signing_key = os.getenv("MCP_JWT_SIGNING_KEY")
    auth = OAuthProxy(
        upstream_authorization_endpoint=f"{cognito_base_url}/oauth2/authorize",
        upstream_token_endpoint=token_url,
        upstream_client_id=client_id,
        upstream_client_secret=client_secret,
        token_verifier=token_verifier,
        base_url=MCP_BASE_URL,
        token_endpoint_auth_method="client_secret_basic",  # Cognito expects Basic Auth
        forward_pkce=True,  # Cognito supports PKCE — keep end-to-end security
        jwt_signing_key=jwt_signing_key,
        client_storage=_build_storage(jwt_signing_key or ""),
    )

    logger.info(
        "Cognito OAuth proxy enabled: pool=%s, region=%s, domain=%s, "
        "base_url=%s, cognito_url=%s",
        pool_id,
        region,
        domain,
        MCP_BASE_URL,
        cognito_base_url,
    )
    return auth


mcp = FastMCP("gemini-websearch", auth=_build_cognito_auth())

# --- Response formatting ---


def _format_response(response) -> dict[str, Any]:
    """Extract text, sources, and inline citations from a Gemini response."""
    result: dict[str, Any] = {"text": response.text}

    if not response.candidates:
        return result

    metadata = response.candidates[0].grounding_metadata
    if not metadata:
        return result

    if metadata.web_search_queries:
        result["search_queries"] = list(metadata.web_search_queries)

    chunks = metadata.grounding_chunks or []
    sources = [
        {"title": chunk.web.title, "uri": chunk.web.uri}
        for chunk in chunks
        if chunk.web
    ]
    if sources:
        result["sources"] = sources

    supports = metadata.grounding_supports or []
    if supports and chunks:
        result["cited_text"] = _build_cited_text(response.text, supports, chunks)

    return result


def _build_cited_text(text: str, supports, chunks) -> str:
    """Insert inline citation links into text, processing from end to start to preserve offsets."""
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


# --- Tools ---

_GOOGLE_SEARCH_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())]
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
    return json.dumps(_format_response(response), indent=2)


@mcp.tool()
async def web_search_custom(query: str, system_instruction: str) -> str:
    """Search the web with a custom system instruction to shape the response.

    Args:
        query: The search query or question to answer.
        system_instruction: Instructions for how to process and present results
            (e.g. "Respond in bullet points", "Focus on technical details").
    """
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=system_instruction,
    )
    response = await _gemini.aio.models.generate_content(
        model=GEMINI_MODEL, contents=query, config=config
    )
    return json.dumps(_format_response(response), indent=2)


# --- Entrypoint ---

if __name__ == "__main__":
    logger.info("Starting Gemini Web Search MCP server (streamable-http)")
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)
