"""Tests for retrieval module."""

import tempfile
from pathlib import Path

from noe_train.retrieval.symbol_index import SymbolIndex


def test_symbol_index_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple Python file
        py_file = Path(tmpdir) / "example.py"
        py_file.write_text("""
class MyClass:
    def my_method(self):
        pass

def standalone_function(x):
    return x + 1
""")

        idx = SymbolIndex()
        count = idx.build(tmpdir)
        assert count > 0

        # Lookup class
        results = idx.lookup("MyClass")
        assert len(results) == 1
        assert results[0].kind == "class"

        # Lookup method
        results = idx.lookup("my_method")
        assert len(results) == 1
        assert results[0].kind == "method"
        assert results[0].parent == "MyClass"

        # Lookup function
        results = idx.lookup("standalone_function")
        assert len(results) >= 1


def test_symbol_index_imports():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "a.py").write_text("from b import foo\n")
        Path(tmpdir, "b.py").write_text("def foo(): pass\n")

        idx = SymbolIndex()
        idx.build(tmpdir)

        assert len(idx.imports) >= 1
        connected = idx.expand_1hop("a.py")
        assert "b.py" in connected
