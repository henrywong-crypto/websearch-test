"""Tests for the Gemini WebSearch MCP server."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from server import _format_response, mcp, web_search, web_search_custom

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
        result = _format_response(resp)
        assert result == {"text": "Hello world"}

    def test_text_only_metadata_none(self):
        resp = _make_response("Hello", metadata=None)
        result = _format_response(resp)
        assert result == {"text": "Hello"}

    def test_web_search_queries(self):
        metadata = SimpleNamespace(
            web_search_queries=["q1", "q2"],
            grounding_chunks=None,
            grounding_supports=None,
        )
        resp = _make_response("answer", metadata=metadata)
        result = _format_response(resp)
        assert result["search_queries"] == ["q1", "q2"]

    def test_grounding_chunks_extracts_sources(self):
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
        result = _format_response(resp)
        assert result["sources"] == [
            {"title": "Site A", "uri": "https://a.com"},
            {"title": "Site B", "uri": "https://b.com"},
        ]
        assert "cited_text" not in result

    def test_supports_with_inline_citations(self):
        chunks = [_make_chunk("Site A", "https://a.com")]
        supports = [_make_support(0, 5, [0])]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp)
        assert "[1](https://a.com)" in result["cited_text"]

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
        result = _format_response(resp)
        cited = result["cited_text"]
        assert "[1](https://a.com)" in cited
        assert "[2](https://b.com)" in cited
        assert cited.index("[1]") < cited.index("[2]")

    def test_support_with_empty_chunk_indices_skipped(self):
        chunks = [_make_chunk("A", "https://a.com")]
        supports = [_make_support(0, 5, [])]
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp)
        assert "cited_text" not in result or result.get("cited_text") == "Hello world"

    def test_chunk_index_out_of_range_skipped(self):
        chunks = [_make_chunk("A", "https://a.com")]
        supports = [_make_support(0, 5, [99])]  # index 99 doesn't exist
        metadata = SimpleNamespace(
            web_search_queries=None,
            grounding_chunks=chunks,
            grounding_supports=supports,
        )
        resp = _make_response("Hello world", metadata=metadata)
        result = _format_response(resp)
        assert "cited_text" not in result or "[" not in result.get("cited_text", "")


# ---------------------------------------------------------------------------
# 2. web_search tool
# ---------------------------------------------------------------------------


class TestWebSearch:
    def test_calls_gemini_and_returns_json(self):
        mock_response = _make_response("search result")

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            raw = web_search("test query")

        result = json.loads(raw)
        assert result["text"] == "search result"

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["contents"] == "test query"

    def test_returns_valid_json_with_expected_keys(self):
        chunks = [_make_chunk("Example", "https://example.com")]
        metadata = SimpleNamespace(
            web_search_queries=["test"],
            grounding_chunks=chunks,
            grounding_supports=None,
        )
        mock_response = _make_response("result text", metadata=metadata)

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            raw = web_search("test")

        result = json.loads(raw)
        assert "text" in result
        assert "search_queries" in result
        assert "sources" in result


# ---------------------------------------------------------------------------
# 3. web_search_custom tool
# ---------------------------------------------------------------------------


class TestWebSearchCustom:
    def test_passes_system_instruction(self):
        mock_response = _make_response("custom result")

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            web_search_custom("query", "Be concise")

        call_kwargs = mock_client.models.generate_content.call_args
        config = call_kwargs.kwargs["config"]
        assert config.system_instruction == "Be concise"

    def test_returns_valid_json(self):
        mock_response = _make_response("custom result")

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            raw = web_search_custom("query", "instructions")

        result = json.loads(raw)
        assert result["text"] == "custom result"


# ---------------------------------------------------------------------------
# 4. MCP server integration
# ---------------------------------------------------------------------------


class TestMCPIntegration:
    @pytest.mark.asyncio
    async def test_server_exposes_both_tools(self):
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "web_search" in tool_names
        assert "web_search_custom" in tool_names

    @pytest.mark.asyncio
    async def test_call_web_search_through_mcp(self):
        mock_response = _make_response("mcp result")

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = await mcp.call_tool("web_search", {"query": "test"})

        parsed = json.loads(result.content[0].text)
        assert parsed["text"] == "mcp result"

    @pytest.mark.asyncio
    async def test_call_web_search_custom_through_mcp(self):
        mock_response = _make_response("custom mcp result")

        with patch("server._gemini") as mock_client:
            mock_client.models.generate_content.return_value = mock_response
            result = await mcp.call_tool(
                "web_search_custom",
                {"query": "test", "system_instruction": "Be brief"},
            )

        parsed = json.loads(result.content[0].text)
        assert parsed["text"] == "custom mcp result"
