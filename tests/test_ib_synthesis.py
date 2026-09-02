import tempfile
import unittest
from pathlib import Path

from benchmark_store import BenchmarkStore
from ib_synthesis import (
    ValuationInputs,
    build_synthesis_report,
    calculate_valuations,
    valuation_table,
    workbook_bytes,
)


def sample_inputs(**overrides):
    values = {
        "ticker": "ACME",
        "company_name": "Acme Holdings",
        "current_price": 75.0,
        "shares_outstanding": 100_000_000,
        "market_cap": 7_500_000_000,
        "eps": 5.0,
        "book_value_per_share": 30.0,
        "roe": 0.18,
        "dividend_per_share": 1.5,
        "growth_rate": 0.05,
        "beta": 1.05,
        "risk_free_rate": 0.04,
        "equity_risk_premium": 0.055,
        "cost_of_debt": 0.05,
        "tax_rate": 0.21,
        "total_debt": 2_000_000_000,
        "cash": 500_000_000,
        "free_cash_flow": 450_000_000,
        "ebitda": 900_000_000,
        "sector_pe": 18.0,
        "sector_ev_ebitda": 10.0,
    }
    values.update(overrides)
    return ValuationInputs(**values)


class ValuationEngineTests(unittest.TestCase):
    def test_runs_all_requested_valuation_methods(self):
        inputs = sample_inputs()
        result = calculate_valuations(inputs)
        self.assertEqual(
            set(result["model_values"]),
            {"DCF", "EV / EBITDA", "P / E", "Dividend discount", "Residual income", "Graham"},
        )
        self.assertGreater(result["fair_value"], 0)
        self.assertLessEqual(result["range_low"], result["range_high"])
        self.assertAlmostEqual(sum(result["weights"].values()), 1.0)
        self.assertFalse(valuation_table(inputs, result).empty)
        self.assertIn("IB-STYLE VALUATION SYNTHESIS", build_synthesis_report(inputs, result))

    def test_dividend_model_is_omitted_without_dividends(self):
        result = calculate_valuations(sample_inputs(dividend_per_share=0.0))
        self.assertEqual(result["model_values"]["Dividend discount"], 0.0)
        self.assertNotIn("Dividend discount", result["active_values"])

    def test_excel_export_is_a_valid_xlsx_archive(self):
        inputs = sample_inputs()
        result = calculate_valuations(inputs)
        self.assertTrue(workbook_bytes(inputs, result).startswith(b"PK"))


class BenchmarkStoreTests(unittest.TestCase):
    def test_persists_reports_notes_and_portfolio(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = BenchmarkStore(Path(temp_dir) / "test.db")
            report_id = store.save_report("IB style synthesis", "ACME", "report", {"fair": 90})
            saved = store.get_report(report_id)
            self.assertEqual(saved["metadata"]["fair"], 90)

            store.add_note("ACME", "Catalyst", "New product cycle")
            self.assertEqual(store.list_notes("ACME")[0]["title"], "Catalyst")

            store.upsert_portfolio("ACME", "Acme", 10, 70, 75, 90, "$")
            portfolio = store.portfolio_frame()
            self.assertEqual(len(portfolio), 1)
            self.assertEqual(portfolio.iloc[0]["Unrealized P/L"], 50)
            store.delete_portfolio("ACME")
            self.assertTrue(store.portfolio_frame().empty)


if __name__ == "__main__":
    unittest.main()
