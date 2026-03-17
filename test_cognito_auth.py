"""Tests for Cognito OAuth configuration via OAuthProxy + JWTVerifier."""

import importlib
import os
import time
from unittest.mock import patch

import pytest
from fastmcp.server.auth import OAuthProxy
from fastmcp.server.auth.providers.jwt import JWTVerifier, RSAKeyPair


# --- Helpers ---

# Base env vars that server.py needs beyond COGNITO_* settings.
_BASE_ENV = {k: v for k, v in os.environ.items() if not k.startswith("COGNITO")}

_COGNITO_FULL_ENV = {
    "COGNITO_USER_POOL_ID": "us-east-1_Test",
    "COGNITO_CLIENT_ID": "client-id",
    "COGNITO_CLIENT_SECRET": "secret",
    "COGNITO_DOMAIN": "https://test.auth.us-east-1.amazoncognito.com",
}


def _reload_server(**cognito_vars):
    """Reload server module with specific COGNITO env vars (others preserved)."""
    env = {**_BASE_ENV, **cognito_vars}
    with patch.dict(os.environ, env, clear=True):
        import server
        importlib.reload(server)
        return server


# --- Configuration tests ---


def test_no_auth_without_env_var():
    """When COGNITO_USER_POOL_ID is not set, _auth should be None."""
    srv = _reload_server()
    assert srv._auth is None


def test_auth_requires_client_id():
    """Missing COGNITO_CLIENT_ID should raise RuntimeError."""
    env = {**_COGNITO_FULL_ENV}
    del env["COGNITO_CLIENT_ID"]
    with pytest.raises(RuntimeError, match="COGNITO_CLIENT_ID"):
        _reload_server(**env)


def test_auth_requires_client_secret():
    """Missing COGNITO_CLIENT_SECRET should raise RuntimeError."""
    env = {**_COGNITO_FULL_ENV}
    del env["COGNITO_CLIENT_SECRET"]
    with pytest.raises(RuntimeError, match="COGNITO_CLIENT_SECRET"):
        _reload_server(**env)


def test_auth_requires_domain():
    """Missing COGNITO_DOMAIN should raise RuntimeError."""
    env = {**_COGNITO_FULL_ENV}
    del env["COGNITO_DOMAIN"]
    with pytest.raises(RuntimeError, match="COGNITO_DOMAIN"):
        _reload_server(**env)


def test_auth_configured_with_all_vars():
    """With all required vars set, _auth should be an OAuthProxy instance."""
    srv = _reload_server(**_COGNITO_FULL_ENV)
    assert isinstance(srv._auth, OAuthProxy)


def test_region_extracted_from_pool_id():
    """Region should be auto-extracted from the pool ID if COGNITO_REGION is not set."""
    env = {**_COGNITO_FULL_ENV, "COGNITO_USER_POOL_ID": "eu-west-1_MyPool"}
    srv = _reload_server(**env)
    assert srv._cognito_region == "eu-west-1"


def test_explicit_region_overrides_pool_id():
    """Explicit COGNITO_REGION should take precedence over the pool ID prefix."""
    env = {**_COGNITO_FULL_ENV, "COGNITO_REGION": "us-west-2"}
    srv = _reload_server(**env)
    assert srv._cognito_region == "us-west-2"


# --- JWTVerifier token validation tests ---

ISSUER = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Test"
AUDIENCE = "test-client-id"


@pytest.fixture
def rsa_keys():
    return RSAKeyPair.generate()


@pytest.fixture
def verifier(rsa_keys):
    return JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
    )


@pytest.mark.asyncio
async def test_valid_token_accepted(verifier, rsa_keys):
    """A properly signed, unexpired token should be accepted."""
    token = rsa_keys.create_token(issuer=ISSUER, audience=AUDIENCE, scopes=["openid"])
    result = await verifier.verify_token(token)
    assert result is not None
    assert result.token == token
    assert "openid" in result.scopes


@pytest.mark.asyncio
async def test_expired_token_rejected(verifier, rsa_keys):
    """An expired token should be rejected."""
    token = rsa_keys.create_token(
        issuer=ISSUER, audience=AUDIENCE, expires_in_seconds=-60,
    )
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_issuer_rejected(verifier, rsa_keys):
    """A token with wrong issuer should be rejected."""
    token = rsa_keys.create_token(issuer="https://wrong-issuer.example.com", audience=AUDIENCE)
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_audience_rejected(verifier, rsa_keys):
    """A token with wrong audience should be rejected."""
    token = rsa_keys.create_token(issuer=ISSUER, audience="wrong-client")
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_wrong_key_rejected(rsa_keys):
    """A token signed with a different key should be rejected."""
    other_keys = RSAKeyPair.generate()
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
    )
    token = other_keys.create_token(issuer=ISSUER, audience=AUDIENCE)
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_garbage_token_rejected(verifier):
    """A completely invalid token string should be rejected."""
    result = await verifier.verify_token("not.a.valid.jwt")
    assert result is None


@pytest.mark.asyncio
async def test_required_scopes_enforced(rsa_keys):
    """A token missing required scopes should be rejected."""
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
        required_scopes=["admin"],
    )
    token = rsa_keys.create_token(issuer=ISSUER, audience=AUDIENCE, scopes=["openid"])
    result = await verifier.verify_token(token)
    assert result is None


@pytest.mark.asyncio
async def test_required_scopes_satisfied(rsa_keys):
    """A token with all required scopes should be accepted."""
    verifier = JWTVerifier(
        public_key=rsa_keys.public_key,
        issuer=ISSUER,
        audience=AUDIENCE,
        required_scopes=["openid", "profile"],
    )
    token = rsa_keys.create_token(issuer=ISSUER, audience=AUDIENCE, scopes=["openid", "profile", "email"])
    result = await verifier.verify_token(token)
    assert result is not None
