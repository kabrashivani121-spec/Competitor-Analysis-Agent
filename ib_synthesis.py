"""Investment-banking style valuation and benchmarking helpers."""

from __future__ import annotations

import importlib.util
import json
import statistics
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

import config


REFERENCE_ENGINE = (
    config.PROJECT_ROOT
    / "02_investment_banking_stock-valuation"
    / "stock_valuation.py"
)


@dataclass
class ValuationInputs:
    ticker: str
    company_name: str
    currency: str = "$"
    current_price: float = 0.0
    shares_outstanding: float = 0.0
    market_cap: float = 0.0
    eps: float = 0.0
    book_value_per_share: float = 0.0
    roe: float = 0.0
    dividend_per_share: float = 0.0
    growth_rate: float = 0.05
    beta: float = 1.0
    risk_free_rate: float = 0.045
    equity_risk_premium: float = 0.055
    cost_of_debt: float = 0.05
    tax_rate: float = 0.21
    total_debt: float = 0.0
    cash: float = 0.0
    free_cash_flow: float = 0.0
    ebitda: float = 0.0
    sector_pe: float = 20.0
    sector_ev_ebitda: float = 10.0
    peer_tickers: list[str] = field(default_factory=list)

    @property
    def net_debt(self) -> float:
        return self.total_debt - self.cash

    @property
    def equity_value(self) -> float:
        if self.market_cap > 0:
            return self.market_cap
        return self.current_price * self.shares_outstanding


def _positive(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number and number > 0 else default
    except (TypeError, ValueError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _bounded_growth(value: float) -> float:
    return min(max(value, -0.10), 0.20)


def calculate_cost_of_capital(inputs: ValuationInputs) -> dict[str, float]:
    """Calculate CAPM cost of equity and capital-structure weighted WACC."""
    cost_of_equity = inputs.risk_free_rate + inputs.beta * inputs.equity_risk_premium
    equity = max(inputs.equity_value, 0.0)
    debt = max(inputs.total_debt, 0.0)
    capital = equity + debt
    if capital <= 0:
        wacc = cost_of_equity
    else:
        wacc = (
            equity / capital * cost_of_equity
            + debt / capital * inputs.cost_of_debt * (1 - inputs.tax_rate)
        )
    return {"cost_of_equity": cost_of_equity, "wacc": wacc}


def _dcf_value(inputs: ValuationInputs, discount_rate: float) -> float:
    if discount_rate <= 0 or inputs.shares_outstanding <= 0:
        return 0.0
    fcf_per_share = inputs.free_cash_flow / inputs.shares_outstanding
    if fcf_per_share <= 0 and inputs.eps > 0:
        payout = min(inputs.dividend_per_share / inputs.eps, 1.0) if inputs.dividend_per_share > 0 else 0.5
        fcf_per_share = inputs.eps * payout
    if fcf_per_share <= 0:
        return 0.0

    forecast_growth = _bounded_growth(inputs.growth_rate)
    terminal_growth = min(max(inputs.growth_rate, 0.0), 0.04)
    if discount_rate - terminal_growth < 0.03:
        terminal_growth = max(discount_rate - 0.03, 0.0)
    if discount_rate <= terminal_growth:
        return 0.0

    years = 5
    pv_cash_flows = sum(
        fcf_per_share * (1 + forecast_growth) ** year / (1 + discount_rate) ** year
        for year in range(1, years + 1)
    )
    terminal_fcf = fcf_per_share * (1 + forecast_growth) ** years
    terminal_value = terminal_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)
    return pv_cash_flows + terminal_value / (1 + discount_rate) ** years


def _pe_value(inputs: ValuationInputs) -> float:
    return inputs.eps * inputs.sector_pe if inputs.eps > 0 and inputs.sector_pe > 0 else 0.0


def _ev_ebitda_value(inputs: ValuationInputs) -> float:
    if (
        inputs.ebitda <= 0
        or inputs.sector_ev_ebitda <= 0
        or inputs.shares_outstanding <= 0
    ):
        return 0.0
    equity_value = inputs.ebitda * inputs.sector_ev_ebitda - inputs.net_debt
    return equity_value / inputs.shares_outstanding if equity_value > 0 else 0.0


def _ddm_value(inputs: ValuationInputs, cost_of_equity: float) -> float:
    if inputs.dividend_per_share <= 0:
        return 0.0
    growth = min(max(inputs.growth_rate, 0.0), 0.10)
    if cost_of_equity - growth < 0.03:
        return 0.0
    return inputs.dividend_per_share * (1 + growth) / (cost_of_equity - growth)


def _residual_income_value(inputs: ValuationInputs, cost_of_equity: float) -> float:
    if inputs.book_value_per_share <= 0 or cost_of_equity <= 0:
        return 0.0
    growth = _bounded_growth(inputs.growth_rate)
    years = 5
    initial_ri = (inputs.roe - cost_of_equity) * inputs.book_value_per_share
    pv_ri = sum(
        initial_ri * (1 + growth) ** year / (1 + cost_of_equity) ** year
        for year in range(1, years + 1)
    )
    terminal_ri = initial_ri * (1 + growth) ** years
    terminal_value = terminal_ri / cost_of_equity if terminal_ri > 0 else 0.0
    return inputs.book_value_per_share + pv_ri + terminal_value / (1 + cost_of_equity) ** years


def _graham_value(inputs: ValuationInputs) -> float:
    if inputs.eps <= 0 or inputs.risk_free_rate <= 0:
        return 0.0
    growth_percent = min(max(inputs.growth_rate, 0.0), 0.20) * 100
    return (
        inputs.eps
        * (8.5 + 2 * growth_percent)
        * 4.4
        / (inputs.risk_free_rate * 100)
    )


MODEL_WEIGHTS = {
    "DCF": 0.25,
    "EV / EBITDA": 0.20,
    "P / E": 0.20,
    "Dividend discount": 0.10,
    "Residual income": 0.15,
    "Graham": 0.10,
}


def calculate_valuations(inputs: ValuationInputs) -> dict[str, Any]:
    """Run all valuation models and synthesize a banker-style fair value range."""
    capital = calculate_cost_of_capital(inputs)
    values = {
        "DCF": _dcf_value(inputs, capital["wacc"]),
        "EV / EBITDA": _ev_ebitda_value(inputs),
        "P / E": _pe_value(inputs),
        "Dividend discount": _ddm_value(inputs, capital["cost_of_equity"]),
        "Residual income": _residual_income_value(inputs, capital["cost_of_equity"]),
        "Graham": _graham_value(inputs),
    }
    available = {name: value for name, value in values.items() if value > 0}

    if len(available) >= 3:
        median = statistics.median(available.values())
        available = {
            name: value
            for name, value in available.items()
            if median / 3 <= value <= median * 3
        }

    weight_total = sum(MODEL_WEIGHTS[name] for name in available)
    normalized_weights = (
        {name: MODEL_WEIGHTS[name] / weight_total for name in available}
        if weight_total
        else {}
    )
    fair_value = sum(available[name] * normalized_weights[name] for name in available)
    sorted_values = sorted(available.values())
    if sorted_values:
        if len(sorted_values) >= 4:
            low = statistics.quantiles(sorted_values, n=4, method="inclusive")[0]
            high = statistics.quantiles(sorted_values, n=4, method="inclusive")[2]
        else:
            low, high = min(sorted_values), max(sorted_values)
    else:
        low = high = 0.0

    upside = (
        (fair_value / inputs.current_price - 1)
        if fair_value > 0 and inputs.current_price > 0
        else 0.0
    )
    return {
        **capital,
        "model_values": values,
        "active_values": available,
        "weights": normalized_weights,
        "fair_value": fair_value,
        "range_low": low,
        "range_high": high,
        "upside": upside,
    }


def valuation_table(inputs: ValuationInputs, result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    range_width = {
        "DCF": 0.15,
        "EV / EBITDA": 0.10,
        "P / E": 0.10,
        "Dividend discount": 0.12,
        "Residual income": 0.12,
        "Graham": 0.15,
    }
    for model, value in result["model_values"].items():
        if value <= 0:
            continue
        spread = range_width[model]
        rows.append(
            {
                "Methodology": model,
                "Low": value * (1 - spread),
                "Midpoint": value,
                "High": value * (1 + spread),
                "Weight": result["weights"].get(model, 0.0),
                "Implied upside": value / inputs.current_price - 1 if inputs.current_price else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_synthesis_report(inputs: ValuationInputs, result: dict[str, Any]) -> str:
    direction = "upside" if result["upside"] >= 0 else "downside"
    active = ", ".join(result["active_values"]) or "No applicable methods"
    report_currency = r"\$" if inputs.currency == "$" else inputs.currency
    price = f"{report_currency}{inputs.current_price:,.2f}"
    fair = f"{report_currency}{result['fair_value']:,.2f}"
    value_range = (
        f"{report_currency}{result['range_low']:,.2f}–"
        f"{report_currency}{result['range_high']:,.2f}"
    )
    return f"""# IB-STYLE VALUATION SYNTHESIS: {inputs.company_name}

- **Ticker:** {inputs.ticker}
- **Current price:** {price}
- **Weighted fair value:** {fair}
- **Selected valuation range:** {value_range}
- **Implied {direction}:** {abs(result['upside']):.1%}

## Executive conclusion

The blended analysis indicates a fair-value midpoint of **{fair}**, implying **{abs(result['upside']):.1%} {direction}** from the current share price. The synthesis uses the methods supported by the available financial data and removes extreme outliers before reweighting the remaining approaches.

## Methodology set

Active methodologies: {active}.

- Cost of equity (CAPM): {result['cost_of_equity']:.1%}
- WACC: {result['wacc']:.1%}
- Sector median P/E: {inputs.sector_pe:.1f}x
- Sector median EV/EBITDA: {inputs.sector_ev_ebitda:.1f}x

## Banker considerations

- DCF and residual-income outputs are most sensitive to growth and discount-rate assumptions.
- Trading-comparable outputs depend on peer selection and the consistency of EBITDA, debt, and share-count units.
- DDM is excluded automatically when dividends are unavailable or the growth/discount spread is unreliable.
- The valuation range should be presented as decision support, not as a single-point prediction.
"""


def _load_reference_engine():
    if not REFERENCE_ENGINE.exists():
        raise FileNotFoundError("The extracted stock-valuation engine is missing")
    spec = importlib.util.spec_from_file_location("reference_stock_valuation", REFERENCE_ENGINE)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load the extracted stock-valuation engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fetch_moex_data(ticker: str) -> dict[str, Any]:
    """Use the extracted repository's MOEX and Smart-Lab adapters without prompts."""
    engine = _load_reference_engine()
    clean_ticker = ticker.upper().replace(".ME", "")
    price = _positive(engine.moex_price(clean_ticker))
    dividends = engine.moex_dividends(clean_ticker)
    recent_dividends = dividends[-4:] if dividends else []
    dps = sum(_positive(value) for _, value in recent_dividends)
    fundamentals = engine.fetch_smartlab(clean_ticker, verbose=False) or {}
    # Smart-Lab exposes share count in millions and statement totals in RUB
    # billions. Normalize them to the raw units used by the common engine.
    shares = _positive(fundamentals.get("shares")) * 1_000_000
    net_debt = _number(fundamentals.get("net_debt")) * 1_000_000_000
    return {
        "ticker": clean_ticker,
        "company_name": engine.moex_name(clean_ticker),
        "currency": "₽",
        "current_price": price,
        "shares_outstanding": shares,
        "market_cap": price * shares,
        "eps": _positive(fundamentals.get("eps")),
        "book_value_per_share": _positive(fundamentals.get("bvps")),
        "roe": _positive(fundamentals.get("roe"), 0.15),
        "dividend_per_share": dps,
        "growth_rate": _number(fundamentals.get("g"), 0.08),
        "beta": _positive(engine.calc_beta_moex(clean_ticker), 1.0),
        "risk_free_rate": 0.16,
        "equity_risk_premium": 0.06,
        "total_debt": max(net_debt, 0.0),
        "cash": max(-net_debt, 0.0),
        "free_cash_flow": _number(fundamentals.get("fcf")) * 1_000_000_000,
        "ebitda": _positive(fundamentals.get("ebitda")) * 1_000_000_000,
        "sector_pe": _positive(engine.get_sector_pe(clean_ticker), 10.5),
        "sector_ev_ebitda": _positive(engine.get_sector_ev_ebitda(clean_ticker), 5.0),
    }


def fetch_yahoo_data(ticker: str) -> dict[str, Any]:
    """Load international public-market inputs from Yahoo Finance."""
    import yfinance as yf

    cache_dir = config.RUNTIME_DIR / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    clean_ticker = ticker.strip().upper()
    security = yf.Ticker(clean_ticker)
    info = security.info or {}
    history = security.history(period="5d", auto_adjust=False)
    current_price = (
        _positive(history["Close"].iloc[-1])
        if not history.empty and "Close" in history
        else _positive(info.get("currentPrice") or info.get("regularMarketPrice"))
    )
    shares = _positive(info.get("sharesOutstanding"))
    market_cap = _positive(info.get("marketCap"), current_price * shares)
    free_cash_flow = _number(info.get("freeCashflow"))
    return {
        "ticker": clean_ticker,
        "company_name": info.get("longName") or info.get("shortName") or clean_ticker,
        "currency": info.get("currency") or "$",
        "current_price": current_price,
        "shares_outstanding": shares,
        "market_cap": market_cap,
        "eps": _positive(info.get("trailingEps") or info.get("forwardEps")),
        "book_value_per_share": _positive(info.get("bookValue")),
        "roe": _positive(info.get("returnOnEquity"), 0.15),
        "dividend_per_share": _positive(
            info.get("trailingAnnualDividendRate") or info.get("dividendRate")
        ),
        "growth_rate": _number(
            info.get("earningsGrowth") or info.get("revenueGrowth"),
            0.05,
        ),
        "beta": _positive(info.get("beta"), 1.0),
        "risk_free_rate": 0.045,
        "equity_risk_premium": 0.055,
        "total_debt": _positive(info.get("totalDebt")),
        "cash": _positive(info.get("totalCash")),
        "free_cash_flow": free_cash_flow,
        "ebitda": _positive(info.get("ebitda")),
        "sector_pe": _positive(info.get("trailingPE"), 20.0),
        "sector_ev_ebitda": _positive(info.get("enterpriseToEbitda"), 10.0),
    }


def fetch_peer_benchmarks(tickers: list[str]) -> tuple[dict[str, float], pd.DataFrame]:
    """Fetch peer multiples, filter extreme observations, and return medians."""
    import yfinance as yf

    cache_dir = config.RUNTIME_DIR / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    rows = []
    for ticker in tickers:
        clean = ticker.strip().upper()
        if not clean:
            continue
        info = yf.Ticker(clean).info or {}
        pe = _positive(info.get("trailingPE") or info.get("forwardPE"))
        ev_ebitda = _positive(info.get("enterpriseToEbitda"))
        if pe or ev_ebitda:
            rows.append({"Ticker": clean, "P / E": pe or None, "EV / EBITDA": ev_ebitda or None})

    frame = pd.DataFrame(rows)
    benchmarks = {"sector_pe": 20.0, "sector_ev_ebitda": 10.0}
    for column, key in (("P / E", "sector_pe"), ("EV / EBITDA", "sector_ev_ebitda")):
        if column not in frame:
            continue
        values = [float(value) for value in frame[column].dropna() if float(value) > 0]
        if not values:
            continue
        median = statistics.median(values)
        filtered = [value for value in values if median / 3 <= value <= median * 3]
        benchmarks[key] = statistics.median(filtered or values)
    return benchmarks, frame


def extract_financials_from_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """Extract valuation inputs from an uploaded annual report using OpenAI."""
    from openai import OpenAI
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("No extractable text was found in the PDF")
    text = text[:60_000]
    client = OpenAI(
        api_key=config.OPENAI_API_KEY,
        timeout=config.OPENAI_TIMEOUT_SECONDS,
        max_retries=config.OPENAI_MAX_RETRIES,
    )
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract public-company valuation inputs from annual-report text. "
                    "Return JSON only with keys: company_name, eps, book_value_per_share, "
                    "roe, dividend_per_share, growth_rate, shares_outstanding, total_debt, "
                    "cash, free_cash_flow, ebitda. Use raw currency units for totals, decimal "
                    "rates for roe and growth_rate, and null for unavailable values."
                ),
            },
            {"role": "user", "content": text},
        ],
    )
    return json.loads(response.choices[0].message.content or "{}")


def workbook_bytes(
    inputs: ValuationInputs,
    result: dict[str, Any],
    portfolio: pd.DataFrame | None = None,
) -> bytes:
    """Create a banker-friendly Excel workbook in memory."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        valuation_table(inputs, result).to_excel(writer, sheet_name="Valuation", index=False)
        pd.DataFrame(
            [
                {"Metric": "Ticker", "Value": inputs.ticker},
                {"Metric": "Current price", "Value": inputs.current_price},
                {"Metric": "Weighted fair value", "Value": result["fair_value"]},
                {"Metric": "Range low", "Value": result["range_low"]},
                {"Metric": "Range high", "Value": result["range_high"]},
                {"Metric": "Cost of equity", "Value": result["cost_of_equity"]},
                {"Metric": "WACC", "Value": result["wacc"]},
            ]
        ).to_excel(writer, sheet_name="Summary", index=False)
        if portfolio is not None and not portfolio.empty:
            portfolio.to_excel(writer, sheet_name="Portfolio", index=False)
    return output.getvalue()
