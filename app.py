"""
Main Streamlit Application for Competitor Analysis System
Multi-agent system powered by CrewAI for automated competitor analysis
"""

import logging
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
from openai import APIConnectionError, APITimeoutError, AuthenticationError, RateLimitError

import config
from benchmark_store import BenchmarkStore
from crewai import Crew, Process
from agents import create_all_agents, create_llm
from ib_synthesis import (
    ValuationInputs,
    build_synthesis_report,
    calculate_valuations,
    extract_financials_from_pdf,
    fetch_moex_data,
    fetch_peer_benchmarks,
    fetch_yahoo_data,
    valuation_table,
    workbook_bytes,
)
from tasks import create_all_tasks
from trusted_sources import build_trusted_source_context
from utils import (
    DETAILED_ANALYSIS_FRAMEWORK,
    PDFReportGenerator,
    competitor_report_issues,
    enforce_quantification_disclosures,
    format_report_for_display,
    extract_key_metrics,
    generate_filename
)

# Configure page
st.set_page_config(
    page_title="Competitor Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = logging.getLogger(__name__)


def describe_analysis_error(exc: Exception) -> str:
    """Translate API/network failures without leaking credentials or internals."""
    if isinstance(exc, AuthenticationError):
        return "OpenAI authentication failed. Verify OPENAI_API_KEY in .env and restart the app."
    if isinstance(exc, RateLimitError):
        return "The OpenAI rate limit or quota was reached. Check account usage, then retry."
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return (
            "The app could not reach OpenAI over HTTPS. Check firewall/network access; "
            "the configured key was not rejected."
        )

    chain = []
    current = exc
    while current is not None and len(chain) < 6:
        chain.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    details = " | ".join(chain).lower()

    if any(marker in details for marker in ("authentication", "invalid_api_key", "401", "unauthorized")):
        return "OpenAI authentication failed. Verify OPENAI_API_KEY in .env and restart the app."
    if any(marker in details for marker in ("rate limit", "ratelimit", "429", "quota")):
        return "The API rate limit or quota was reached. Check account usage, then retry."
    if any(marker in details for marker in ("connection", "connecterror", "winerror 10013", "timeout")):
        return (
            "The app could not reach an external API over HTTPS. Check firewall/network access; "
            "the configured keys were not rejected."
        )
    return f"Analysis failed: {type(exc).__name__}: {exc}"


def synthesize_report_from_evidence(
    evidence_context: str,
    company_name: str,
    industry: str,
    num_competitors: int,
    prior_issues: list[str],
) -> str:
    """Request a standalone report directly from the LLM, outside an agent loop."""
    prompt = f"""
You are the final report formatter for a trusted competitive-intelligence workflow.
Using only the evidence and corrections below, return one complete standalone Markdown report.
Output the report itself only—no preface, no completion note, and no reference to text above.

The first line must be exactly: # COMPETITOR ANALYSIS REPORT: {company_name}
Use the current report date: {datetime.now().strftime('%Y-%m-%d')}.
Include these exact H2 headings:
## EXECUTIVE SUMMARY
## KEY FINDINGS
## COMPETITIVE LANDSCAPE OVERVIEW
## DETAILED COMPETITOR ANALYSIS
## COMPETITIVE COMPARISON MATRIX
## MARKET OPPORTUNITIES
## COMPETITIVE THREATS
## STRATEGIC RECOMMENDATIONS
## CONCLUSION
## TRUSTED SOURCE REGISTER

Under DETAILED COMPETITOR ANALYSIS, include at least {num_competitors} named H3
competitor profiles and use this exact three-stage framework:
- **ASSESS**: **Industry Overview** and **Competitor Landscape**.
- **BENCHMARK (Competitor Deep-Dive)**: for every H3 competitor profile include
  **Financial Performance**, **Business Model**, **Product/Services**, **Pricing Structure**,
  **Brand & Marketing**, **Sales & Distribution**, **Market Reach/Share**, **Customer Perception**,
  **Operational Capabilities**, **Talent & Culture**, **Strategic Moves**, and **SWOT Analysis**.
- **STRATEGIZE**: **Competitive Strategy** and **Actionable Recommendations**.
Keep the stage names and framework fields inside the DETAILED COMPETITOR ANALYSIS H2 section.
Use bold labels for stages and fields; reserve H3 headings only for competitor names.
Consulting quantification standard: for Financial Performance, Product/Services, Pricing Structure,
Sales & Distribution, and Market Reach/Share, include at least one trusted, cited magnitude wherever
available. Examples include revenue, growth, profit/margin, price/range, product or model count,
category/revenue mix, unit deliveries, distributor/dealer/store count, countries or regions served,
and market share. Every number must include its unit or currency, reporting period/date, and source.
If no trusted magnitude is available, write exactly "Unknown / not verified" for that field.
KEY FINDINGS must contain numbered items. Apply every correction in
the quality review. Preserve URLs from the evidence. Mark unsupported facts
"Unknown / not verified" and distinguish facts from analyst inference. Do not invent sources,
dates, prices, statistics, or customer sentiment.

The agent-loop response was rejected for: {', '.join(prior_issues)}.

EVIDENCE AND REVIEW MATERIAL:
{evidence_context}
"""
    response = create_llm(temperature=0.1).invoke(prompt)
    content = getattr(response, "content", response)
    return str(content).strip()


def synthesize_complete_report(
    tasks: list,
    company_name: str,
    industry: str,
    num_competitors: int,
    prior_issues: list[str],
) -> str:
    """Collect task outputs and bypass agent-loop final-answer shorthand."""
    context_sections = []
    labels = ["Research", "Market analysis", "Product benchmark", "Quality review"]
    for label, task in zip(labels, tasks[:-1]):
        if task.output and task.output.raw_output:
            context_sections.append(f"## {label} input\n{task.output.raw_output[:24000]}")
    return synthesize_report_from_evidence(
        "\n\n".join(context_sections),
        company_name,
        industry,
        num_competitors,
        prior_issues,
    )


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'report_content' not in st.session_state:
        st.session_state.report_content = None
    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    if 'industry' not in st.session_state:
        st.session_state.industry = ""
    if 'analysis_running' not in st.session_state:
        st.session_state.analysis_running = False
    if 'ib_prefill' not in st.session_state:
        st.session_state.ib_prefill = {}
    if 'ib_result' not in st.session_state:
        st.session_state.ib_result = None
    if 'peer_frame' not in st.session_state:
        st.session_state.peer_frame = pd.DataFrame()


@st.cache_resource
def get_benchmark_store() -> BenchmarkStore:
    """Return the local history, notes, and portfolio store."""
    return BenchmarkStore()


def validate_api_keys():
    """Validate that required API keys are configured"""
    is_valid, message = config.validate_config()
    
    if not is_valid:
        st.error(f"⚠️ Configuration Error: {message}")
        st.info("""
        **Setup Instructions:**
        1. Create a `.env` file in the project directory
        2. Add your API keys:
           ```
           OPENAI_API_KEY=your_openai_key_here
           SERPAPI_API_KEY=your_serpapi_key_here
           ```
        3. Restart the application
        
        **Get API Keys:**
        - OpenAI: https://platform.openai.com/api-keys
        - SerpAPI: https://serpapi.com/manage-api-key
        """)
        return False
    
    return True


def render_analysis_mode_selector() -> str:
    """Let the user switch between the two integrated methodologies."""
    with st.sidebar:
        st.title("Benchmarking Studio")
        mode = st.radio(
            "Analysis model",
            options=["Competitive intelligence", "IB style synthesis"],
            help="Choose the consulting research workflow or public-equity valuation workflow.",
        )
        st.caption(
            "Competitive intelligence produces market, product, pricing, SWOT, and strategy analysis. "
            "IB style synthesis blends intrinsic and trading-comparable valuations."
        )
        st.divider()
    return mode


def render_sidebar():
    """Render sidebar with input form"""
    with st.sidebar:
        st.header("🔍 Competitive Intelligence")

        workflow = st.radio(
            "Workflow",
            ["Company benchmarking", "Product idea discovery"],
            help="Analyze an established company or start from a new product/business idea.",
        )
        idea_mode = workflow == "Product idea discovery"
        
        # Company Information
        st.subheader("Company Information")
        
        company_name = st.text_input(
            "Idea / project name *" if idea_mode else "Company Name *",
            placeholder="e.g., AI procurement copilot" if idea_mode else "e.g., Slack, Shopify, Tesla",
            help="Name this discovery project" if idea_mode else "Enter the company you want to analyze",
        )

        our_product = st.text_area(
            "Product idea *" if idea_mode else "Product or offering (optional)",
            placeholder=(
                "Describe the problem, target users, proposed solution, value proposition, and assumptions"
                if idea_mode
                else "Describe your product, customer, differentiators, or roadmap constraints"
            ),
            help="Adds a specific baseline for the feature, packaging, and gap analysis.",
        )
        
        industry = st.selectbox(
            "Industry *",
            options=config.INDUSTRIES,
            help="Select the industry sector"
        )
        
        st.markdown("---")
        
        # Analysis Settings
        st.subheader("Analysis Settings")
        
        num_competitors = st.slider(
            "Number of Competitors",
            min_value=1,
            max_value=config.MAX_COMPETITORS,
            value=config.DEFAULT_COMPETITORS,
            help="How many competitors to analyze"
        )
        
        analysis_depth = st.selectbox(
            "Analysis Depth",
            options=list(config.ANALYSIS_DEPTH_CONFIG.keys()),
            format_func=lambda x: f"{x.title()} - {config.ANALYSIS_DEPTH_CONFIG[x]['description']}",
            help="Choose the depth of analysis"
        )

        st.markdown("---")
        st.subheader("Trusted research sources")
        st.caption(
            "Web discovery is restricted to official company and regulator sites plus "
            "recognized investment-bank, ratings, industry-research, consulting, and business "
            "publishers. Add source links or upload reports you are authorized to use."
        )
        trusted_urls_text = st.text_area(
            "Official or trusted report URLs",
            placeholder=(
                "One URL per line, such as an annual report, investor-relations page, "
                "initiating-coverage report, or industry outlook"
            ),
            help=(
                "Every listed domain is explicitly approved for this analysis. "
                "Only public pages are fetched; access controls are never bypassed."
            ),
        )
        trusted_files = st.file_uploader(
            "Upload coverage or industry reports",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload public or licensed reports you are permitted to analyze (8 MB per file recommended).",
        )
        with st.expander("Source policy"):
            st.markdown(
                """
                Accepted source classes include official company and competitor websites and
                investor-relations reports; SEC and other regulatory filings; recognized exchanges;
                established investment-bank research; ratings agencies; leading consulting and
                industry-research firms; and established business press. Unrecognized blogs,
                forums, SEO aggregators, and unsourced summaries are excluded.

                Paywalled or licensed research must be supplied by the user; this application does
                not bypass subscriptions, logins, or publisher access controls.
                """
            )
        
        st.markdown("---")
        
        # Action Button
        start_analysis = st.button(
            "🚀 Start Analysis",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.analysis_running
        )
        
        # Validation
        if start_analysis:
            if not company_name or not industry or (idea_mode and not our_product.strip()):
                st.error("Please fill in all required fields (*)")
                return None, None, None, None, None, None, None, None

            trusted_urls = [
                line.strip()
                for line in trusted_urls_text.splitlines()
                if line.strip()
            ]
            uploaded_reports = [(item.name, item.getvalue()) for item in trusted_files]
            return (
                company_name,
                industry,
                num_competitors,
                analysis_depth,
                our_product,
                workflow,
                trusted_urls,
                uploaded_reports,
            )
        
        # Info section
        if st.session_state.analysis_running:
            st.info("⏳ Analysis in progress...")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AI-powered system uses five specialized stages to:
        - 🔎 Research competitors
        - 📊 Analyze market positioning
        - 🧩 Compare product, pricing, and business models
        - ✅ Review evidence quality
        - 📝 Generate strategic recommendations
        """)
        
        return None, None, None, None, None, None, None, None


def render_welcome_screen():
    """Render welcome screen before analysis"""
    st.title("🔍 AI-Powered Competitor Analysis System")
    
    st.markdown("""
    ### Welcome to the Competitor Analysis Platform
    
    This system uses advanced AI agents to conduct comprehensive competitor research and analysis.
    
    #### How It Works:
    
    1. **Research Agent** 🔎
       - Discovers and gathers data on your competitors
       - Collects pricing, features, and customer feedback
       - Identifies market positioning
    
    2. **Analysis Agent** 📊
       - Performs SWOT analysis on each competitor
       - Creates competitive comparison matrices
       - Analyzes market dynamics and trends
    
    3. **Product Agent** 🧩
       - Builds feature, pricing, packaging, and business-model matrices
       - Finds product gaps, switching costs, and unmet customer needs

    4. **Quality Reviewer** ✅
       - Scores source quality, recency, completeness, and actionability
       - Flags unsupported claims and prescribes corrections

    5. **Report Agent** 📝
       - Synthesizes insights into strategic recommendations
       - Generates executive-ready reports
       - Identifies opportunities and threats
    
    #### Get Started:
    
    👈 Fill in the form in the sidebar to begin your competitor analysis.
    
    ---
    """)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Accurate")
        st.write("Real-time data from reliable sources")
    
    with col2:
        st.markdown("### ⚡ Fast")
        st.write("Complete analysis in minutes")
    
    with col3:
        st.markdown("### 💡 Actionable")
        st.write("Strategic recommendations you can use")


def run_competitor_analysis(
    company_name: str,
    industry: str,
    num_competitors: int,
    analysis_depth: str,
    our_product: str = "",
    workflow: str = "Company benchmarking",
    trusted_urls: list[str] | None = None,
    uploaded_reports: list[tuple[str, bytes]] | None = None,
):
    """
    Execute the competitor analysis using CrewAI
    
    Args:
        company_name: Name of the company to analyze
        industry: Industry sector
        num_competitors: Number of competitors to analyze
        analysis_depth: Depth of analysis (quick/standard/deep)
    """
    try:
        st.session_state.analysis_running = True
        st.session_state.company_name = company_name
        st.session_state.industry = industry
        
        # Create progress container
        progress_container = st.container()
        
        with progress_container:
            st.markdown("### 🔄 Analysis in Progress")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Load explicit source material and initialize agents.
            status_text.text("📚 Loading and validating trusted sources...")
            progress_bar.progress(10)

            trusted_source_context, trusted_domains, source_errors = build_trusted_source_context(
                trusted_urls or [],
                uploaded_reports or [],
            )
            for source_error in source_errors:
                st.warning(f"Skipped source: {source_error}")

            status_text.text("🤖 Initializing AI agents with the trusted-source policy...")
            agents = create_all_agents(
                company_name,
                industry,
                analysis_depth,
                our_product,
                trusted_domains,
            )
            logger.info("Agents created successfully")
            
            # Step 2: Create Tasks
            status_text.text("📋 Creating analysis tasks...")
            progress_bar.progress(20)
            
            store = get_benchmark_store()
            prior_parts = []
            for item in store.list_reports(mode="Competitive intelligence", limit=50):
                if item["subject"].casefold() == company_name.casefold():
                    previous = store.get_report(item["id"])
                    if previous:
                        prior_parts.append(previous["content"][:5000])
                    break
            notes = store.list_notes(company_name)
            if notes:
                prior_parts.append(
                    "Manual engagement notes:\n"
                    + "\n".join(f"- {note['title']}: {note['content']}" for note in notes[:10])
                )
            prior_context = "\n\n".join(prior_parts)

            tasks = create_all_tasks(
                agents,
                company_name,
                industry,
                num_competitors,
                analysis_depth,
                our_product,
                prior_context,
                trusted_source_context,
            )
            logger.info("Tasks created successfully")
            
            # Step 3: Create Crew
            status_text.text("👥 Assembling crew...")
            progress_bar.progress(30)
            
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            logger.info("Crew assembled successfully")
            
            # Step 4: Execute Analysis
            status_text.text("🔎 Research Agent: Discovering competitors...")
            progress_bar.progress(40)
            
            # Execute the crew
            with st.spinner("Analysis in progress... This may take several minutes."):
                result = crew.kickoff()

            report_content = enforce_quantification_disclosures(str(result))
            report_issues = competitor_report_issues(report_content, num_competitors)
            if report_issues:
                logger.warning("Incomplete final report; retrying final synthesis: %s", report_issues)
                status_text.text("📝 Final report was incomplete; regenerating the full report...")
                progress_bar.progress(90)
                report_content = enforce_quantification_disclosures(synthesize_complete_report(
                    tasks,
                    company_name,
                    industry,
                    num_competitors,
                    report_issues,
                ))
                report_issues = competitor_report_issues(report_content, num_competitors)
                if report_issues:
                    logger.warning("Quantification/structure gaps remain; applying corrective pass: %s", report_issues)
                    report_content = enforce_quantification_disclosures(synthesize_report_from_evidence(
                        report_content,
                        company_name,
                        industry,
                        num_competitors,
                        report_issues,
                    ))
                    report_issues = competitor_report_issues(report_content, num_competitors)
                if report_issues:
                    raise RuntimeError(
                        "The report generator returned incomplete output after an automatic retry: "
                        + "; ".join(report_issues)
                    )
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Store results
            st.session_state.report_content = report_content
            st.session_state.analysis_complete = True
            st.session_state.analysis_running = False
            get_benchmark_store().save_report(
                mode="Competitive intelligence",
                subject=company_name,
                content=report_content,
                metadata={
                    "industry": industry,
                    "competitors": num_competitors,
                    "depth": analysis_depth,
                    "product_context": our_product,
                    "workflow": workflow,
                    "incremental_context_used": bool(prior_context),
                    "trusted_source_urls": len(trusted_urls or []),
                    "uploaded_reports": len(uploaded_reports or []),
                    "trusted_domains": trusted_domains,
                },
            )
            
            logger.info("Analysis completed successfully")
            
            # Success message
            st.success("🎉 Competitor analysis completed successfully!")
            st.balloons()
            
            # Rerun to show results
            st.rerun()
            
    except Exception as e:
        st.session_state.analysis_running = False
        logger.exception("Error during analysis")
        st.error(f"❌ {describe_analysis_error(e)}")
        st.info("Your inputs are preserved. Correct the connection or credential issue and retry.")


def render_results():
    """Render analysis results in tabs"""
    if not st.session_state.analysis_complete or not st.session_state.report_content:
        return
    
    st.title(f"📊 Competitor Analysis: {st.session_state.company_name}")
    st.markdown(f"**Industry:** {st.session_state.industry} | **Date:** {datetime.now().strftime('%B %d, %Y')}")
    st.markdown("---")
    
    # Extract metrics for summary
    metrics = extract_key_metrics(st.session_state.report_content)
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Competitors Analyzed", metrics.get('num_competitors', 'N/A'))
    
    with col2:
        st.metric("Key Findings", len(metrics.get('key_findings', [])))
    
    with col3:
        st.metric("Opportunities", metrics.get('opportunities', 'N/A'))
    
    with col4:
        st.metric("Threat Level", metrics.get('threat_level', 'Medium'))
    
    st.markdown("---")
    
    # Parse report into sections
    sections = format_report_for_display(st.session_state.report_content)
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Overview",
        "🔍 Detailed Analysis",
        "📊 Competitor Matrix",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.markdown("### Executive Summary")
        if sections['overview']:
            st.markdown(sections['overview'])
        else:
            st.info("Overview section not available")
    
    with tab2:
        st.markdown("### Detailed Competitor Analysis")
        st.markdown("#### Competitive Market Analysis Framework")
        assess_col, benchmark_col, strategize_col = st.columns([1, 2, 1])
        framework_columns = zip(
            (assess_col, benchmark_col, strategize_col),
            DETAILED_ANALYSIS_FRAMEWORK.items(),
        )
        for column, (stage, dimensions) in framework_columns:
            with column:
                with st.container(border=True):
                    st.markdown(f"**{stage}**")
                    for dimension in dimensions:
                        st.markdown(f"- {dimension}")
        st.divider()
        if sections['detailed_analysis']:
            st.markdown(sections['detailed_analysis'])
        else:
            st.info("Detailed analysis section not available")
    
    with tab3:
        st.markdown("### Competitive Comparison Matrix")
        if sections['competitor_matrix']:
            st.markdown(sections['competitor_matrix'])
        else:
            st.info("Comparison matrix not available")
    
    with tab4:
        st.markdown("### Strategic Recommendations")
        if sections['recommendations']:
            st.markdown(sections['recommendations'])
        else:
            st.info("Recommendations section not available")
    
    # Export section
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        # PDF Export
        if st.button("📄 Download PDF", use_container_width=True):
            try:
                with st.spinner("Generating PDF..."):
                    pdf_generator = PDFReportGenerator(
                        st.session_state.company_name,
                        st.session_state.industry
                    )
                    pdf_buffer = pdf_generator.generate_pdf(st.session_state.report_content)
                    
                    filename = generate_filename(st.session_state.company_name, "pdf")
                    
                    st.download_button(
                        label="💾 Save PDF",
                        data=pdf_buffer,
                        file_name=filename,
                        mime="application/pdf"
                    )
                    st.success("PDF generated successfully!")
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
    
    with col2:
        # Text Export
        filename = generate_filename(st.session_state.company_name, "txt")
        st.download_button(
            label="📝 Download Text",
            data=st.session_state.report_content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )
    
    # Reset button
    st.markdown("---")
    if st.button("🔄 Start New Analysis", use_container_width=False):
        st.session_state.analysis_complete = False
        st.session_state.report_content = None
        st.rerun()


def open_saved_competitive_report(saved):
    """Load a saved report before Streamlit renders the next dashboard frame."""
    st.session_state.company_name = saved["subject"]
    st.session_state.industry = saved.get("metadata", {}).get("industry", "")
    st.session_state.report_content = saved["content"]
    st.session_state.analysis_complete = True


def render_competitive_workspace():
    """Render consulting-style report history and working notes."""
    store = get_benchmark_store()
    with st.expander("🗂️ Saved reports and engagement notes"):
        history_tab, notes_tab = st.tabs(["Report history", "Notes"])
        with history_tab:
            reports = store.list_reports(mode="Competitive intelligence")
            if not reports:
                st.caption("Completed competitive-intelligence reports will be stored here.")
            else:
                options = {
                    f"#{item['id']} · {item['subject']} · {item['created_at']}": item["id"]
                    for item in reports
                }
                selected = st.selectbox("Saved report", options, key="ci_saved_report")
                saved = store.get_report(options[selected])
                if saved:
                    st.button(
                        "Open saved report in dashboard",
                        key=f"open_ci_{saved['id']}",
                        on_click=open_saved_competitive_report,
                        args=(saved,),
                    )
                    st.markdown(saved["content"])
                    st.download_button(
                        "Download saved report",
                        saved["content"],
                        file_name=f"{saved['subject']}_competitive_intelligence.md",
                        mime="text/markdown",
                    )
                    if st.button("Delete saved report", key=f"delete_ci_{saved['id']}"):
                        store.delete_report(saved["id"])
                        st.success("Saved report deleted.")
                        st.rerun()
        with notes_tab:
            with st.form("ci_note_form", clear_on_submit=True):
                note_subject = st.text_input(
                    "Subject",
                    value=st.session_state.company_name,
                    placeholder="Company or engagement",
                )
                note_title = st.text_input("Note title")
                note_content = st.text_area("Note")
                save_note = st.form_submit_button("Save note")
            if save_note:
                if note_subject.strip() and note_title.strip() and note_content.strip():
                    store.add_note(note_subject, note_title, note_content)
                    st.success("Note saved.")
                    st.rerun()
                else:
                    st.warning("Complete the subject, title, and note fields.")
            notes = store.list_notes(note_subject.strip() or None)
            for note in notes:
                st.markdown(f"**{note['title']}** · {note['created_at']}")
                st.write(note["content"])


def _ib_value(key: str, default):
    value = st.session_state.ib_prefill.get(key, default)
    return default if value is None else value


def _load_ib_market_data(source: str, ticker: str):
    if not ticker.strip():
        st.warning("Enter a ticker before loading market data.")
        return
    try:
        with st.spinner(f"Loading {ticker.upper()} market and financial data..."):
            loaded = fetch_yahoo_data(ticker) if source == "Yahoo Finance" else fetch_moex_data(ticker)
        st.session_state.ib_prefill = {**st.session_state.ib_prefill, **loaded}
        st.success("Market data loaded. Review every input before running the valuation.")
        st.rerun()
    except Exception as exc:
        logger.exception("Unable to load public-market data")
        st.error(f"Could not load market data: {exc}")


def _render_ib_result():
    payload = st.session_state.ib_result
    if not payload:
        return
    inputs: ValuationInputs = payload["inputs"]
    result = payload["result"]
    report = payload["report"]
    table = valuation_table(inputs, result)

    st.divider()
    st.subheader(f"Valuation synthesis · {inputs.company_name}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current price", f"{inputs.currency}{inputs.current_price:,.2f}")
    col2.metric(
        "Weighted fair value",
        f"{inputs.currency}{result['fair_value']:,.2f}",
        delta=f"{result['upside']:.1%}",
    )
    col3.metric("Range low", f"{inputs.currency}{result['range_low']:,.2f}")
    col4.metric("Range high", f"{inputs.currency}{result['range_high']:,.2f}")

    if table.empty:
        st.warning("No valuation methodology had enough valid inputs. Add financial data and try again.")
        return

    display = table.copy()
    for column in ["Low", "Midpoint", "High"]:
        display[column] = display[column].map(lambda value: f"{inputs.currency}{value:,.2f}")
    display["Weight"] = display["Weight"].map(lambda value: f"{value:.1%}")
    display["Implied upside"] = display["Implied upside"].map(lambda value: f"{value:.1%}")
    st.dataframe(display, hide_index=True, use_container_width=True)
    st.caption("Football-field view: methodology ranges and midpoints")
    ranges = alt.Chart(table).mark_rule(strokeWidth=5, color="#4C78A8").encode(
        x=alt.X("Low:Q", title=f"Value per share ({inputs.currency})"),
        x2="High:Q",
        y=alt.Y("Methodology:N", sort=None, title=None),
        tooltip=["Methodology:N", "Low:Q", "Midpoint:Q", "High:Q"],
    )
    midpoints = alt.Chart(table).mark_point(size=90, filled=True, color="#F58518").encode(
        x="Midpoint:Q",
        y=alt.Y("Methodology:N", sort=None),
        tooltip=["Methodology:N", "Midpoint:Q"],
    )
    st.altair_chart(ranges + midpoints, use_container_width=True)

    with st.expander("View banker-style synthesis", expanded=True):
        st.markdown(report)

    portfolio = get_benchmark_store().portfolio_frame()
    export_col1, export_col2 = st.columns(2)
    export_col1.download_button(
        "Download Markdown",
        report,
        file_name=f"{inputs.ticker or 'company'}_ib_synthesis.md",
        mime="text/markdown",
        use_container_width=True,
    )
    export_col2.download_button(
        "Download Excel workbook",
        workbook_bytes(inputs, result, portfolio),
        file_name=f"{inputs.ticker or 'company'}_valuation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("Add or update portfolio position"):
        with st.form("portfolio_position"):
            shares_owned = st.number_input("Shares owned", min_value=0.0, value=0.0)
            cost_basis = st.number_input(
                "Cost basis per share",
                min_value=0.0,
                value=float(inputs.current_price),
            )
            add_position = st.form_submit_button("Save position")
        if add_position:
            get_benchmark_store().upsert_portfolio(
                inputs.ticker or inputs.company_name,
                inputs.company_name,
                shares_owned,
                cost_basis,
                inputs.current_price,
                result["fair_value"],
                inputs.currency,
            )
            st.success("Portfolio position saved.")


def render_ib_valuation_tab():
    """Collect source data and run the integrated valuation engine."""
    st.markdown(
        "Use manual inputs, prefill from Yahoo Finance or MOEX/Smart-Lab, calculate peer medians, "
        "and optionally extract financial fields from an annual report."
    )
    source_col, ticker_col, button_col = st.columns([1.3, 1, 0.8], vertical_alignment="bottom")
    with source_col:
        data_source = st.radio(
            "Input source",
            ["Manual input", "Yahoo Finance", "MOEX / Smart-Lab"],
            horizontal=True,
            key="ib_data_source",
        )
    with ticker_col:
        market_ticker = st.text_input(
            "Ticker",
            value=str(_ib_value("ticker", "")),
            placeholder="MSFT or SBER",
            key="ib_market_ticker",
        )
    with button_col:
        if st.button(
            "Load market data",
            disabled=data_source == "Manual input",
            use_container_width=True,
        ):
            _load_ib_market_data(data_source, market_ticker)

    with st.expander("Peer trading comparables and annual-report extraction"):
        peers = st.text_input(
            "Yahoo peer tickers",
            placeholder="GOOGL, META, ORCL",
            help="The median P/E and EV/EBITDA of valid observations will prefill the valuation.",
        )
        if st.button("Fetch peer medians"):
            peer_tickers = [item.strip() for item in peers.split(",") if item.strip()]
            if not peer_tickers:
                st.warning("Enter at least one peer ticker.")
            else:
                try:
                    with st.spinner("Fetching trading comparables..."):
                        benchmarks, frame = fetch_peer_benchmarks(peer_tickers)
                    st.session_state.ib_prefill = {**st.session_state.ib_prefill, **benchmarks}
                    st.session_state.peer_frame = frame
                    st.success("Peer medians loaded into the assumptions below.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not fetch peer data: {exc}")
        if not st.session_state.peer_frame.empty:
            st.dataframe(st.session_state.peer_frame, hide_index=True, use_container_width=True)

        annual_report = st.file_uploader("Annual report PDF", type=["pdf"])
        if st.button("Extract financial inputs with OpenAI"):
            if annual_report is None:
                st.warning("Upload a PDF first.")
            elif not config.OPENAI_API_KEY:
                st.error("OPENAI_API_KEY is required for annual-report extraction.")
            else:
                try:
                    with st.spinner("Extracting financial inputs from the report..."):
                        extracted = extract_financials_from_pdf(annual_report.getvalue())
                    clean = {key: value for key, value in extracted.items() if value is not None}
                    st.session_state.ib_prefill = {**st.session_state.ib_prefill, **clean}
                    st.success("Financial inputs extracted. Review the values and their units.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not extract the report: {exc}")

    with st.form("ib_valuation_inputs"):
        st.subheader("Company and market assumptions")
        col1, col2, col3 = st.columns(3)
        ticker = col1.text_input("Ticker / identifier", value=str(_ib_value("ticker", market_ticker)))
        company_name = col2.text_input("Company name", value=str(_ib_value("company_name", ticker or "Company")))
        currency = col3.text_input("Currency symbol", value=str(_ib_value("currency", "$")))

        col1, col2, col3 = st.columns(3)
        current_price = col1.number_input("Current share price", min_value=0.0, value=float(_ib_value("current_price", 0.0)))
        shares_outstanding = col2.number_input(
            "Shares outstanding (raw shares)", min_value=0.0, value=float(_ib_value("shares_outstanding", 0.0)), format="%.0f"
        )
        market_cap = col3.number_input("Market capitalization", min_value=0.0, value=float(_ib_value("market_cap", 0.0)))

        with st.expander("Earnings, cash flow, balance sheet, and dividends", expanded=True):
            col1, col2, col3 = st.columns(3)
            eps = col1.number_input("EPS", value=float(_ib_value("eps", 0.0)))
            book_value = col2.number_input("Book value per share", value=float(_ib_value("book_value_per_share", 0.0)))
            roe_pct = col3.number_input("ROE (%)", value=float(_ib_value("roe", 0.15)) * 100)
            col1, col2, col3 = st.columns(3)
            dividend = col1.number_input("Dividend per share", min_value=0.0, value=float(_ib_value("dividend_per_share", 0.0)))
            free_cash_flow = col2.number_input("Free cash flow (total)", value=float(_ib_value("free_cash_flow", 0.0)), format="%.0f")
            ebitda = col3.number_input("EBITDA (total)", value=float(_ib_value("ebitda", 0.0)), format="%.0f")
            col1, col2 = st.columns(2)
            total_debt = col1.number_input("Total debt", min_value=0.0, value=float(_ib_value("total_debt", 0.0)), format="%.0f")
            cash = col2.number_input("Cash and equivalents", min_value=0.0, value=float(_ib_value("cash", 0.0)), format="%.0f")

        with st.expander("Growth, capital costs, and peer multiples", expanded=True):
            col1, col2, col3 = st.columns(3)
            growth_pct = col1.number_input("Forecast growth (%)", value=float(_ib_value("growth_rate", 0.05)) * 100)
            beta = col2.number_input("Equity beta", min_value=0.0, value=float(_ib_value("beta", 1.0)))
            risk_free_pct = col3.number_input("Risk-free rate (%)", min_value=0.0, value=float(_ib_value("risk_free_rate", 0.045)) * 100)
            col1, col2, col3 = st.columns(3)
            erp_pct = col1.number_input("Equity risk premium (%)", min_value=0.0, value=float(_ib_value("equity_risk_premium", 0.055)) * 100)
            debt_cost_pct = col2.number_input("Pre-tax cost of debt (%)", min_value=0.0, value=float(_ib_value("cost_of_debt", 0.05)) * 100)
            tax_pct = col3.number_input("Tax rate (%)", min_value=0.0, max_value=100.0, value=float(_ib_value("tax_rate", 0.21)) * 100)
            col1, col2 = st.columns(2)
            sector_pe = col1.number_input("Peer median P/E", min_value=0.0, value=float(_ib_value("sector_pe", 20.0)))
            sector_ev_ebitda = col2.number_input("Peer median EV/EBITDA", min_value=0.0, value=float(_ib_value("sector_ev_ebitda", 10.0)))

        run_valuation = st.form_submit_button("Run IB synthesis", type="primary", use_container_width=True)

    if run_valuation:
        inputs = ValuationInputs(
            ticker=ticker.strip().upper(), company_name=company_name.strip() or ticker.strip().upper() or "Company",
            currency=currency or "$", current_price=current_price, shares_outstanding=shares_outstanding,
            market_cap=market_cap, eps=eps, book_value_per_share=book_value, roe=roe_pct / 100,
            dividend_per_share=dividend, growth_rate=growth_pct / 100, beta=beta,
            risk_free_rate=risk_free_pct / 100, equity_risk_premium=erp_pct / 100,
            cost_of_debt=debt_cost_pct / 100, tax_rate=tax_pct / 100, total_debt=total_debt,
            cash=cash, free_cash_flow=free_cash_flow, ebitda=ebitda,
            sector_pe=sector_pe, sector_ev_ebitda=sector_ev_ebitda,
            peer_tickers=[item.strip().upper() for item in peers.split(",") if item.strip()],
        )
        result = calculate_valuations(inputs)
        report = build_synthesis_report(inputs, result)
        st.session_state.ib_result = {"inputs": inputs, "result": result, "report": report}
        get_benchmark_store().save_report(
            "IB style synthesis", inputs.ticker or inputs.company_name, report,
            {"company_name": inputs.company_name, "current_price": inputs.current_price, "fair_value": result["fair_value"]},
        )
        st.success("Valuation complete and saved to history.")

    _render_ib_result()


def render_ib_portfolio_tab():
    store = get_benchmark_store()
    frame = store.portfolio_frame()
    if frame.empty:
        st.info("Add a position from a completed valuation to create a portfolio benchmark.")
        return
    col1, col2, col3 = st.columns(3)
    col1.metric("Market value", f"{frame['Market value'].sum():,.2f}")
    col2.metric("Cost value", f"{frame['Cost value'].sum():,.2f}")
    col3.metric("Unrealized P/L", f"{frame['Unrealized P/L'].sum():,.2f}")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    st.download_button("Download portfolio CSV", frame.to_csv(index=False), "portfolio.csv", "text/csv")
    ticker = st.selectbox("Position to remove", frame["ticker"].tolist())
    if st.button("Remove selected position"):
        store.delete_portfolio(ticker)
        st.success(f"Removed {ticker}.")
        st.rerun()


def render_ib_history_tab():
    reports = get_benchmark_store().list_reports(mode="IB style synthesis")
    if not reports:
        st.info("Completed valuation syntheses will be stored here.")
        return
    options = {
        f"#{item['id']} · {item['subject']} · {item['created_at']}": item["id"]
        for item in reports
    }
    selected = st.selectbox("Saved synthesis", options, key="ib_saved_report")
    saved = get_benchmark_store().get_report(options[selected])
    if saved:
        st.markdown(saved["content"])
        st.download_button("Download saved synthesis", saved["content"], f"{saved['subject']}_ib_synthesis.md", "text/markdown")
        if st.button("Delete saved synthesis", key=f"delete_ib_{saved['id']}"):
            get_benchmark_store().delete_report(saved["id"])
            st.success("Saved synthesis deleted.")
            st.rerun()


def render_ib_synthesis():
    st.title("🏦 IB Style Synthesis")
    st.caption(
        "DCF, trading comparables, dividend discount, residual income, Graham valuation, "
        "outlier filtering, weighted synthesis, portfolio benchmarking, and workbook export."
    )
    valuation_tab, portfolio_tab, methodology_tab, history_tab = st.tabs(
        ["Valuation", "Portfolio", "Methodology", "History"]
    )
    with valuation_tab:
        render_ib_valuation_tab()
    with portfolio_tab:
        render_ib_portfolio_tab()
    with methodology_tab:
        st.markdown(
            """
            ### Integrated methodology

            - **Intrinsic value:** five-year DCF using WACC, a Gordon-growth dividend model when valid,
              and a residual-income model anchored to book value and ROE.
            - **Trading comparables:** P/E and EV/EBITDA against user assumptions or fetched peer medians.
            - **Cross-check:** Graham earnings-growth valuation.
            - **Synthesis:** extreme outputs outside one-third to three times the median are removed;
              the remaining methodologies are reweighted into a fair-value midpoint and interquartile range.
            - **Data sources:** manual inputs, Yahoo Finance, MOEX/Smart-Lab, and OpenAI-assisted annual-report extraction.

            Review accounting units and peer comparability before relying on any output. This is analytical
            decision support, not investment advice.
            """
        )
    with history_tab:
        render_ib_history_tab()


def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    mode = render_analysis_mode_selector()
    if mode == "IB style synthesis":
        render_ib_synthesis()
        return

    # Competitive intelligence requires both LLM and search credentials.
    if not validate_api_keys():
        return

    (
        company_name,
        industry,
        num_competitors,
        analysis_depth,
        our_product,
        workflow,
        trusted_urls,
        uploaded_reports,
    ) = render_sidebar()
    
    # Main content area
    if st.session_state.analysis_complete:
        # Show results
        render_results()
    elif company_name and industry:
        # Run analysis
        run_competitor_analysis(
            company_name,
            industry,
            num_competitors,
            analysis_depth,
            our_product,
            workflow,
            trusted_urls,
            uploaded_reports,
        )
    else:
        # Show welcome screen
        render_welcome_screen()
    render_competitive_workspace()


if __name__ == "__main__":
    main()
