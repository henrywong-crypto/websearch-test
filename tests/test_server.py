"""Tests for the Gemini WebSearch MCP server."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from server import _format_response, mcp, web_search

# ---------------------------------------------------------------------------
# Helpers to build mock Gemini response objects
# ---------------------------------------------------------------------------


def _make_chunk(title: str, uri: str) -> SimpleNamespace:
    return SimpleNamespace(web=SimpleNamespace(title=title, uri=uri))


def _make_support(start: int, end: int, chunk_indices: list[int]) -> SimpleNamespace:
    return SimpleNamespace(
        segment=SimpleNamespace(start_index=start, end_index=end),
        grounding_chunk_indices=chunk_indices,
    )


def _make_response(
    text: str,
    *,
    metadata=None,
) -> SimpleNamespace:
    """Build a minimal mock Gemini response."""
    candidate = SimpleNamespace(grounding_metadata=metadata)
    return SimpleNamespace(text=text, candidates=[candidate])


# ---------------------------------------------------------------------------
# 1. _format_response  (pure logic)
# ---------------------------------------------------------------------------


class TestFormatResponse:
    def test_text_only_no_metadata(self):
        resp = _make_response("Hello world")
        assert _format_response(resp, "test") == "Hello world"

    def test_text_only_metadata_none(self):
        resp = _make_response("Hello", metadata=None)
        assert _format_response(resp, "test") == "Hello"

    def test_no_grounding_just_text(self):
        metadata = SimpleNamespace(
            web_search_queries=["q1", "q2"],
            grounding_chunks=None,
            grounding_supports=None,
        )
        resp = _make_response("answer", metadata=metadata)
        result = _format_response(resp, "test")
        assert "answer" in result
        assert "No links found." in result

    def test_links_and_reminder_included(self):
        chunks = [
            _make_chunk("Site A", "https://a.com"),
            _make_chunk("Site B", "https://b.com"),
        ]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        resp = _make_response("info", metadata=metadata)
        result = _format_response(resp, "my query")
        assert result.startswith('Web search results for query: "my query"')
        assert '"title": "Site A"' in result
        assert '"url": "https://a.com"' in result
        assert '"title": "Site B"' in result
        assert "info" in result
        assert "REMINDER:" in result

    def test_inline_citations(self):
        chunks = [_make_chunk("Site A", "https://a.com")]
        supports = [_make_support(0, 5, [0])]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp, "test")
        assert "[1](https://a.com)" in result

    def test_multiple_supports_descending_order(self):
        text = "First sentence. Second sentence."
        chunks = [
            _make_chunk("A", "https://a.com"),
            _make_chunk("B", "https://b.com"),
        ]
        supports = [
            _make_support(0, 15, [0]),  # "First sentence."
            _make_support(16, 32, [1]),  # "Second sentence."
        ]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response(text, metadata=metadata)
        result = _format_response(resp, "test")
        assert "[1](https://a.com)" in result
        assert "[2](https://b.com)" in result
        assert result.index("[1]") < result.index("[2]")

    def test_empty_chunk_indices_skipped(self):
        chunks = [_make_chunk("A", "https://a.com")]
        supports = [_make_support(0, 5, [])]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp, "test")
        assert "Hello world" in result

    def test_chunk_index_out_of_range_skipped(self):
        chunks = [_make_chunk("A", "https://a.com")]
        supports = [_make_support(0, 5, [99])]  # index 99 doesn't exist
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp, "test")
        assert "Hello world" in result

    def test_empty_candidates(self):
        resp = SimpleNamespace(text="Hello", candidates=[])
        assert _format_response(resp, "test") == "Hello"

    def test_no_links_says_no_links(self):
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=[],
            grounding_supports=None,
        )
        resp = _make_response("answer", metadata=metadata)
        result = _format_response(resp, "test")
        assert "No links found." in result


# ---------------------------------------------------------------------------
# 2. web_search tool
# ---------------------------------------------------------------------------


class TestWebSearch:
    @pytest.mark.asyncio
    async def test_calls_gemini_and_returns_envelope(self):
        mock_response = _make_response("search result")

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await web_search("test query")

        # No metadata → falls through to plain text
        assert result == "search result"

        call_kwargs = mock_client.aio.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "test query"

    @pytest.mark.asyncio
    async def test_returns_envelope_with_links(self):
        chunks = [_make_chunk("Example", "https://example.com")]
        metadata = SimpleNamespace(
            web_search_queries=["test"],
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        mock_response = _make_response("result text", metadata=metadata)

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await web_search("test")

        assert 'Web search results for query: "test"' in result
        assert "result text" in result
        assert '"title": "Example"' in result
        assert "REMINDER:" in result


# ---------------------------------------------------------------------------
# 3. MCP server integration
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    @pytest.mark.asyncio
    async def test_server_exposes_web_search_tool(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names

    @pytest.mark.asyncio
    async def test_call_web_search_through_mcp(self):
        mock_response = _make_response("mcp result")

        with patch("server._gemini") as mock_client:
            mock_client.aio.models.generate_content = AsyncMock(
                return_value=mock_response
            )
            result = await mcp.call_tool("web_search", {"query": "test"})

        assert result.content[0].text == "mcp result"
