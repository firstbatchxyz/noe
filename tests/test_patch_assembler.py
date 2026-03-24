"""Tests for patch assembler."""

from noe_train.sandbox.patch_assembler import PatchAssembly, PatchNACK


def test_patch_assembly_add_hunks():
    assembly = PatchAssembly()
    assembly.add_hunk("a.py", 0, "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new")
    assembly.add_hunk("b.py", 0, "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y")

    assert len(assembly.files) == 2
    assert "a.py" in assembly.hunks
    assert "b.py" in assembly.hunks


def test_patch_assembly_to_unified():
    assembly = PatchAssembly()
    hunk1 = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new"
    hunk2 = "--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y"
    assembly.add_hunk("a.py", 0, hunk1)
    assembly.add_hunk("b.py", 0, hunk2)

    unified = assembly.to_unified_diff()
    assert "a.py" in unified
    assert "b.py" in unified


def test_patch_assembly_finalized_guard():
    assembly = PatchAssembly()
    assembly.finalized = True

    try:
        assembly.add_hunk("c.py", 0, "diff")
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_patch_nack_structure():
    nack = PatchNACK(
        file_path="broken.py",
        hunk_idx=2,
        error="context mismatch",
        suggestion="regenerate hunk 2",
    )
    assert nack.file_path == "broken.py"
    assert nack.hunk_idx == 2
