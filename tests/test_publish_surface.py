from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublishSurfaceTest(unittest.TestCase):
    def test_required_paths_exist(self) -> None:
        required = [
            ROOT / "README.md",
            ROOT / "resources" / "paper_inventory.json",
            ROOT / "config" / "deterministic_execution.json",
            ROOT / "docs" / "runtime_envelope.md",
            ROOT / "lock" / "requirements-exact.txt",
            ROOT / "lock" / "python-runtime.lock.json",
            ROOT / "repr" / "README.md",
            ROOT / "repr" / "reproduce.py",
            ROOT / "repr" / "expected.json",
            ROOT / "repr" / "requirements.txt",
            ROOT / "repr" / "_lib" / "methods.py",
            ROOT / "scripts" / "run_artifact_parity.py",
            ROOT / "scripts" / "generate_runtime_lock.py",
            ROOT / "scripts" / "verify_runtime_lock.py",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing required path: {path}")

    def test_repository_has_no_nested_publish_tree(self) -> None:
        old_root = ROOT / ("pub" + "lish")
        self.assertFalse(old_root.exists(), f"obsolete nested repository root: {old_root}")

    def test_no_internal_process_terms(self) -> None:
        forbidden_patterns = (
            re.compile("/" + "Users/"),
            re.compile("file:" + "//"),
            re.compile(r"paper[\s_-]*b\b", re.IGNORECASE),
            re.compile(r"chine" + r"se[\s_-]*wall", re.IGNORECASE),
            re.compile(r"(?:^|[/\\])filter_cp_" + r"repr(?:[/\\]|$)", re.IGNORECASE),
        )
        publishable_roots = (
            ROOT / "artifacts",
            ROOT / "config",
            ROOT / "docs",
            ROOT / "lock",
            ROOT / "repr",
            ROOT / "resources",
            ROOT / "scripts",
            ROOT / "src",
            ROOT / "tests",
        )
        text_suffixes = {".csv", ".json", ".md", ".py", ".txt"}
        paths = [ROOT / "README.md", ROOT / ".gitignore"]
        for root in publishable_roots:
            paths.extend(path for path in root.rglob("*") if path.is_file())

        for path in paths:
            if path.suffix not in text_suffixes and path.name != ".gitignore":
                continue
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                self.assertIsNone(pattern.search(text), f"forbidden term in {path}: {pattern.pattern}")


if __name__ == "__main__":
    unittest.main()
