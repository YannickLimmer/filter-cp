from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "repr" / "reproduce.py"


class ReproduceStructureTest(unittest.TestCase):
    def _module_ast(self) -> ast.Module:
        src = REPRO.read_text(encoding="utf-8")
        return ast.parse(src)

    def test_method_order_is_defined_and_unique(self) -> None:
        tree = self._module_ast()
        method_order = None
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "METHOD_ORDER":
                        method_order = ast.literal_eval(node.value)
        self.assertIsNotNone(method_order, "METHOD_ORDER not found")
        self.assertIsInstance(method_order, list)
        self.assertEqual(len(method_order), len(set(method_order)))
        self.assertGreaterEqual(len(method_order), 10)

    def test_cell_aliases_exist(self) -> None:
        src = REPRO.read_text(encoding="utf-8")
        for alias in ("all", "primary", "hero", "gpvar_scale"):
            self.assertIn(f"'{alias}'", src)


if __name__ == "__main__":
    unittest.main()
