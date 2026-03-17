"""MCP server that provides web search via Gemini Grounding with Google Search."""

import base64
import json
import os
import logging
import ssl

from typing import Any

import certifi
import httpx
from fastmcp import FastMCP
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.providers.jwt import JWTVerifier
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Build a custom SSL context that works with corporate proxies / custom CAs.
_ca_file = os.environ.get("SSL_CERT_FILE") or os.path.join(os.path.dirname(__file__), "ca-bundle.pem")
if not os.path.exists(_ca_file):
    _ca_file = certifi.where()

# Force all env vars so that urllib3/requests/google-auth also pick up the CA bundle.
os.environ.setdefault("SSL_CERT_FILE", _ca_file)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_file)

# Patch certifi to return our CA bundle (google-auth uses certifi.where() internally).
certifi.where = lambda: _ca_file

_ssl_ctx = ssl.create_default_context(cafile=_ca_file)
# Python 3.14 / OpenSSL 3.5 enables VERIFY_X509_STRICT by default, which rejects
# certificates whose Basic Constraints extension is not marked critical (e.g. Netskope
# proxy CAs).  Relax that check so corporate proxy CA bundles work.
_ssl_ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT
logger.info(f"SSL CA file: {_ca_file}, loaded: {_ssl_ctx.cert_store_stats()}")


# --- Auth configuration ---
_cognito_pool_id = os.getenv("COGNITO_USER_POOL_ID")
_mcp_host = os.getenv("MCP_HOST", "0.0.0.0")
_mcp_port = int(os.getenv("MCP_PORT", "8080"))

# Public base URL that MCP clients (e.g. Cursor) reach the server at.
# Must be HTTPS in production so OAuth redirect URIs are accepted.
# Example: https://mcp.example.com  (no trailing slash)
_mcp_base_url = os.getenv("MCP_BASE_URL", f"http://{_mcp_host}:{_mcp_port}")

_auth: OAuthProxy | None = None

if _cognito_pool_id:
    _cognito_region = os.getenv("COGNITO_REGION") or _cognito_pool_id.split("_")[0]
    _cognito_client_id = os.getenv("COGNITO_CLIENT_ID", "")
    _cognito_client_secret = os.getenv("COGNITO_CLIENT_SECRET", "")
    _cognito_domain = os.getenv("COGNITO_DOMAIN", "")  # domain prefix only, e.g. "myapp"

    if not _cognito_client_id:
        raise RuntimeError("COGNITO_CLIENT_ID must be set when COGNITO_USER_POOL_ID is configured")
    if not _cognito_client_secret:
        raise RuntimeError("COGNITO_CLIENT_SECRET must be set when COGNITO_USER_POOL_ID is configured")
    if not _cognito_domain:
        raise RuntimeError("COGNITO_DOMAIN must be set (domain prefix, e.g. 'myapp')")

    _cognito_base_url = f"https://{_cognito_domain}.auth.{_cognito_region}.amazoncognito.com"

    _issuer_url = f"https://cognito-idp.{_cognito_region}.amazonaws.com/{_cognito_pool_id}"
    _jwks_uri = f"{_issuer_url}/.well-known/jwks.json"

    # Cognito requires client credentials in the POST body for token exchange.
    # The audience for Cognito access tokens is the User Pool client ID.
    _cognito_scopes_raw = os.getenv("COGNITO_SCOPES", "openid profile email")
    _cognito_scopes = [s.strip() for s in _cognito_scopes_raw.split() if s.strip()]

    _token_verifier = JWTVerifier(
        jwks_uri=_jwks_uri,
        issuer=_issuer_url,
        # NOTE: Do NOT set audience here. Cognito access tokens use `client_id`
        # instead of the standard `aud` claim, so JWTVerifier's audience check
        # would always fail. The issuer check is sufficient to validate the token
        # came from the correct Cognito user pool.
        required_scopes=_cognito_scopes,
        http_client=httpx.AsyncClient(verify=_ssl_ctx),
    )

    # Work around authlib not injecting client credentials into the token exchange
    # request. Patch httpx to add Basic Auth for Cognito token requests, matching
    # the approach in ~/cognito (Authorization: Basic base64(client_id:client_secret)).
    _basic_creds = base64.b64encode(
        f"{_cognito_client_id}:{_cognito_client_secret}".encode()
    ).decode()
    _cognito_token_url = f"{_cognito_base_url}/oauth2/token"

    _original_httpx_send = httpx.AsyncClient.send
    async def _inject_cognito_auth(self, request, **kwargs):
        if str(request.url) == _cognito_token_url:
            request.headers["authorization"] = f"Basic {_basic_creds}"
        return await _original_httpx_send(self, request, **kwargs)
    httpx.AsyncClient.send = _inject_cognito_auth

    _auth = OAuthProxy(
        upstream_authorization_endpoint=f"{_cognito_base_url}/oauth2/authorize",
        upstream_token_endpoint=_cognito_token_url,
        upstream_client_id=_cognito_client_id,
        upstream_client_secret=_cognito_client_secret,
        token_verifier=_token_verifier,
        base_url=_mcp_base_url,
        # Cognito expects client credentials as Basic Auth header.
        token_endpoint_auth_method="client_secret_basic",
        # Cognito supports PKCE — keep end-to-end security.
        forward_pkce=True,
        jwt_signing_key=os.getenv("MCP_JWT_SIGNING_KEY"),
    )

    logger.info(
        f"Cognito OAuth proxy enabled: pool={_cognito_pool_id}, region={_cognito_region}, "
        f"domain={_cognito_domain}, base_url={_mcp_base_url}, cognito_url={_cognito_base_url}"
    )
else:
    logger.info("No COGNITO_USER_POOL_ID set — running without auth")

mcp = FastMCP(
    "gemini-websearch",
    auth=_auth,
)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _get_client() -> genai.Client:
    http_options = types.HttpOptions(httpx_client=httpx.Client(verify=_ssl_ctx))

    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project:
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        return genai.Client(
            vertexai=True, project=project, location=location,
            http_options=http_options,
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key, http_options=http_options)

    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT (for Vertex AI) or GEMINI_API_KEY (for API key auth)"
    )


def _format_response(response) -> dict[str, Any]:
    """Extract text, citations, and grounding metadata from a Gemini response."""
    candidate = response.candidates[0]
    result: dict[str, Any] = {"text": response.text}

    metadata = candidate.grounding_metadata
    if not metadata:
        return result

    if metadata.web_search_queries:
        result["search_queries"] = list(metadata.web_search_queries)

    chunks = metadata.grounding_chunks or []
    sources = []
    for chunk in chunks:
        if chunk.web:
            sources.append({"title": chunk.web.title, "uri": chunk.web.uri})
    if sources:
        result["sources"] = sources

    supports = metadata.grounding_supports or []
    if supports and chunks:
        cited_text = response.text
        sorted_supports = sorted(
            supports, key=lambda s: s.segment.end_index, reverse=True
        )
        for support in sorted_supports:
            end_index = support.segment.end_index
            if not support.grounding_chunk_indices:
                continue
            links = []
            for i in support.grounding_chunk_indices:
                if i < len(chunks) and chunks[i].web:
                    links.append(f"[{i + 1}]({chunks[i].web.uri})")
            if links:
                cited_text = (
                    cited_text[:end_index]
                    + " " + ", ".join(links)
                    + cited_text[end_index:]
                )
        result["cited_text"] = cited_text

    return result


@mcp.tool()
def web_search(query: str) -> str:
    """Search the web using Gemini Grounding with Google Search.

    Returns up-to-date information with citations from real web sources.
    Use this for any question that benefits from current, factual information.

    Args:
        query: The search query or question to answer.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=query, config=config
    )
    return json.dumps(_format_response(response), indent=2)


@mcp.tool()
def web_search_custom(query: str, system_instruction: str) -> str:
    """Search the web with a custom system instruction to shape the response.

    Args:
        query: The search query or question to answer.
        system_instruction: Instructions for how to process and present results
            (e.g. "Respond in bullet points", "Focus on technical details").
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())],
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=query, config=config
    )
    return json.dumps(_format_response(response), indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        query = sys.argv[2] if len(sys.argv) > 2 else "hello world"
        print(f"Testing Gemini API directly with query: {query!r}")
        try:
            client = _get_client()
            print("Client created OK")
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
            response = client.models.generate_content(
                model=GEMINI_MODEL, contents=query, config=config
            )
            print(json.dumps(_format_response(response), indent=2))
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        logger.info("Starting Gemini Web Search MCP server (streamable-http)")
        mcp.run(transport="streamable-http", host=_mcp_host, port=_mcp_port)
