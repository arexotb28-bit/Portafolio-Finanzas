from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "herramienta_finanzas_completa.py"
if not MODULE_PATH.exists():
    MODULE_PATH = PROJECT_ROOT / "final.py"
EXCEL_PATH = PROJECT_ROOT / "base_diaria.xlsx"
if not EXCEL_PATH.exists():
    EXCEL_PATH = PROJECT_ROOT / "final-base_diaria.xlsx"


def load_module():
    """Importa la aplicación de trabajo sin ejecutar la interfaz Streamlit."""
    spec = importlib.util.spec_from_file_location("portfolio_lab", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FinancialEngineTests(unittest.TestCase):
    """Valida carga, invalidación y salidas financieras principales."""

    @classmethod
    def setUpClass(cls):
        cls.app = load_module()
        cls.data = cls.app.load_market_data(EXCEL_PATH)

    def test_excel_loads_all_sources(self):
        self.assertFalse(self.data.prices.empty)
        self.assertFalse(self.data.benchmark.empty)
        self.assertFalse(self.data.risk_free.empty)
        self.assertFalse(self.data.metadata.empty)

    def test_fingerprint_changes_with_file_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "base_diaria.xlsx"
            shutil.copy2(EXCEL_PATH, copied)
            before = self.app.file_fingerprint(copied)
            with copied.open("ab") as stream:
                stream.write(b"version")
            after = self.app.file_fingerprint(copied)
            self.assertNotEqual(before, after)

    def test_ols_exposes_requested_diagnostics(self):
        result1 = self.app.run_modulo1(self.data)
        assets = [column for column in result1["returns"].columns if column != self.app.BENCHMARK_TICKER]
        regression = self.app.ols_against_benchmark(
            result1["returns"][assets[0]],
            result1["returns"][self.app.BENCHMARK_TICKER],
            result1["rf_period"],
            result1["annual_factor"],
        )
        required = {
            "R2 ajustado",
            "Error estandar alpha",
            "Error estandar beta",
            "p-value alpha",
            "p-value beta",
            "SSE",
            "Durbin-Watson",
        }
        self.assertTrue(required.issubset(regression))
        self.assertTrue(np.isfinite(regression["Beta OLS"]))

    def test_clean_outputs_use_relative_destination(self):
        """Comprueba que la exportación limpia crea cuatro archivos reproducibles."""
        result1 = self.app.run_modulo1(self.data)
        with tempfile.TemporaryDirectory() as temp_dir:
            created = self.app.save_clean_outputs(
                self.data,
                result1["indicators"],
                Path(temp_dir),
            )
            self.assertEqual(len(created), 4)
            self.assertTrue(all(path.exists() for path in created))


if __name__ == "__main__":
    unittest.main()
