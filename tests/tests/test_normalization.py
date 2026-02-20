from dataclasses import dataclass

from graph_rlm.backend.src.mcp_integration.utils import normalize_mcp_result


@dataclass
class TextContent:
    type: str
    text: str


def test_normalization():
    print("Testing MCP result normalization...")

    # Case 1: Simple list with TextContent-like object
    mock_result = [TextContent(type="text", text="Paper summary here")]
    normalized = normalize_mcp_result(mock_result)
    print(f"Case 1 (Object): {normalized}")
    assert normalized == "Paper summary here"

    # Case 2: Dict representing TextContent
    mock_result_dict = [{"type": "text", "text": "Dict result here"}]
    normalized_dict = normalize_mcp_result(mock_result_dict)
    print(f"Case 2 (Dict): {normalized_dict}")
    assert normalized_dict == "Dict result here"

    # Case 3: Multiple items (should NOT normalize)
    mock_multi = ["item1", "item2"]
    normalized_multi = normalize_mcp_result(mock_multi)
    print(f"Case 3 (Multi): {normalized_multi}")
    assert normalized_multi == mock_multi

    # Case 4: Non-list (should NOT normalize)
    mock_str = "just a string"
    normalized_str = normalize_mcp_result(mock_str)
    print(f"Case 4 (String): {normalized_str}")
    assert normalized_str == mock_str

    print("✓ MCP result normalization test passed!")


if __name__ == "__main__":
    test_normalization()
