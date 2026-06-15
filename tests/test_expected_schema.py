from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = ROOT / "repr" / "expected.json"


class ExpectedSchemaTest(unittest.TestCase):
    def test_expected_json_exists(self) -> None:
        self.assertTrue(EXPECTED.exists(), f"missing file: {EXPECTED}")

    def test_expected_json_schema(self) -> None:
        data = json.loads(EXPECTED.read_text(encoding="utf-8"))
        for key in (
            "alpha",
            "n_seeds_real",
            "table_hero",
            "table_scale_gpvar",
            "table_logvol_main",
            "table_diag_main",
        ):
            self.assertIn(key, data)

        self.assertGreater(data["alpha"], 0.0)
        self.assertLess(data["alpha"], 1.0)
        self.assertGreaterEqual(int(data["n_seeds_real"]), 1)

        for table_key in ("table_hero", "table_scale_gpvar"):
            table = data[table_key]
            self.assertIsInstance(table, dict)
            self.assertGreater(len(table), 0)
            for _, record in table.items():
                self.assertIn("N", record)
                self.assertGreater(int(record["N"]), 0)
                for method, val in record.items():
                    if method == "N":
                        continue
                    self.assertIsInstance(val, list)
                    self.assertEqual(len(val), 2)
                    self.assertIsInstance(val[0], (int, float))
                    self.assertIsInstance(val[1], (int, float))


if __name__ == "__main__":
    unittest.main()
