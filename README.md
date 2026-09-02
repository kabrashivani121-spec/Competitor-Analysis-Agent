# Benchmarking Studio

A Streamlit application with two selectable analysis models:

- **Competitive intelligence** — consulting-style market, competitor, product, pricing, SWOT, evidence-quality, and strategic-recommendation analysis using a five-stage CrewAI workflow.
- **IB style synthesis** — public-equity valuation using DCF, EV/EBITDA, P/E, dividend discount, residual income, and Graham cross-checks, blended into a banker-style fair-value range.

The implementation adapts methodology from [WwaitW/competitor-intel](https://github.com/WwaitW/competitor-intel) and [S0uLL6/Stock-valuation-programm](https://github.com/S0uLL6/Stock-valuation-programm) within the existing OpenAI/SerpAPI application. Optional local clones of those upstream repositories are ignored by Git and are not required to run this integrated app.

## Features

### Competitive intelligence

- Competitor discovery and multi-source web research
- Trusted-source gate that excludes unrecognized search results and retains source classifications
- Dedicated discovery for initiating-coverage, equity-research, industry-outlook, annual-report, and regulatory-filing sources
- Public URL ingestion and PDF/text/Markdown uploads for reports the user is authorized to use
- Company, product, feature, pricing, packaging, and business-model comparison
- Customer sentiment, target segment, hiring, investment, and momentum signals
- SWOT, positioning, market trends, white-space analysis, threats, and opportunities
- Dedicated product-benchmarking stage
- Product-idea discovery workflow using a problem, target user, solution, value proposition, and assumptions
- Evidence-quality gate scoring source quality, recency, completeness, consistency, depth, and actionability; low-scoring dimensions trigger supplemental research
- Executive report with immediate, short-term, and long-term recommendations
- PDF/text export, local report history, and engagement notes
- Incremental knowledge reuse: prior reports and manual notes are injected into repeat analyses to focus on material changes
- Quick, standard, and deep research configurations

### Trusted-source policy

Competitive-intelligence web results are limited to official company and competitor websites;
official regulatory filings and recognized exchanges; established investment-bank, ratings,
consulting, and industry-research publishers; recognized business press; and reputable review
platforms. The research and quality-review stages must mark unsupported facts as
`Unknown / not verified`, and the final report includes a trusted source register.

The sidebar also accepts direct public report URLs and uploaded PDF, text, or Markdown reports.
Direct links are treated as explicitly user-approved sources only after a safe public-URL check.
Paywalls, logins, and publisher access controls are never bypassed; licensed reports must be
uploaded by an authorized user.

### IB style synthesis

- Manual assumptions or prefilling from Yahoo Finance and MOEX/Smart-Lab
- Yahoo peer ticker collection with filtered median P/E and EV/EBITDA
- OpenAI-assisted financial extraction from annual-report PDFs
- CAPM cost of equity and capital-structure weighted WACC
- DCF, P/E, EV/EBITDA, dividend discount, residual-income, and Graham methodologies
- Automatic omission of inapplicable methods and extreme-outlier filtering
- Reweighted fair-value midpoint, valuation range, upside/downside, and football-field chart
- Markdown and Excel workbook export
- Persistent valuation history and portfolio benchmarking

## Project layout

```text
app.py                                  Streamlit UI and model selector
agents.py / tasks.py                    Five-stage competitive-intelligence workflow
tools.py                                SerpAPI search and data-processing tools
trusted_sources.py                      Source allowlist, safe report ingestion, and text extraction
ib_synthesis.py                         Valuation, market-data, PDF, and export engine
benchmark_store.py                      SQLite reports, notes, and portfolio storage
config.py / utils.py                    Configuration, reporting, and utility helpers
tests/                                  Unit and Streamlit mode tests
```

## Setup on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set credentials in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
SERPAPI_API_KEY=your_serpapi_api_key
OPENAI_MODEL=gpt-4.1-mini
```

Both keys are required for competitive intelligence. IB calculations and manual inputs work without them; `OPENAI_API_KEY` is only required there for annual-report extraction. Yahoo Finance and MOEX/Smart-Lab do not require API keys.

## Run

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501), then choose **Competitive intelligence** or **IB style synthesis** in the sidebar.

Runtime state is stored under `.runtime/`, including `benchmarking.db` and the Yahoo Finance cache. `.env`, `.runtime/`, logs, and the virtual environment are ignored by Git.

## Test

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

Valuation output is analytical decision support and is not investment advice. Review source quality, accounting units, peer selection, and assumptions before relying on results.
