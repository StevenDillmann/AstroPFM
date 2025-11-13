"""Test basic package imports."""
import astropfm


def test_version_exists():
    """Test that version attribute exists."""
    assert hasattr(astropfm, "__version__")
    assert isinstance(astropfm.__version__, str)