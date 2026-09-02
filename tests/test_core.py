import json
import unittest
from unittest.mock import patch

from tools import (
    CompetitorSearchTool,
    CoverageReportSearchTool,
    DataProcessorTool,
    OfficialWebsiteSearchTool,
    SearchAPIError,
    execute_serpapi_search,
    infer_official_website,
)
from utils import (
    PDFReportGenerator,
    competitor_report_issues,
    enforce_quantification_disclosures,
    extract_key_metrics,
    format_report_for_display,
)


SAMPLE_REPORT = """# COMPETITOR ANALYSIS REPORT: Acme
## EXECUTIVE SUMMARY
Acme competes in a crowded market.
## KEY FINDINGS
1. Finding one
2. Finding two
## DETAILED COMPETITOR ANALYSIS
**ASSESS**
- **Industry Overview**: Stable demand.
- **Competitor Landscape**: Two direct rivals.
**BENCHMARK (Competitor Deep-Dive)**
### Alpha
- **Financial Performance**: Revenue was $10 million in 2025. (Source: Official report)
- **Business Model**: Direct sales.
- **Product/Services**: 3 product categories in 2025. (Source: Official catalog)
- **Pricing Structure**: $99 per month as of 2025. (Source: Official pricing)
- **Brand & Marketing**: Enterprise brand.
- **Sales & Distribution**: 20 distributors across 4 countries in 2025. (Source: Annual report)
- **Market Reach/Share**: 12% market share in 2025. (Source: Industry outlook)
- **Customer Perception**: Positive.
- **Operational Capabilities**: Scaled operations.
- **Talent & Culture**: Specialist team.
- **Strategic Moves**: New launch.
- **SWOT Analysis**: Strong brand; narrow portfolio.
- **Competitive Threat Level**: High
### Beta
- **Financial Performance**: Unknown / not verified.
- **Business Model**: Subscription.
- **Product/Services**: 2 service categories in 2025. (Source: Official catalog)
- **Pricing Structure**: $49 per month as of 2025. (Source: Official pricing)
- **Brand & Marketing**: Challenger brand.
- **Sales & Distribution**: 8 partners across 2 countries in 2025. (Source: Annual report)
- **Market Reach/Share**: 4% market share in 2025. (Source: Industry outlook)
- **Customer Perception**: Neutral.
- **Operational Capabilities**: Regional operations.
- **Talent & Culture**: Growth team.
- **Strategic Moves**: Partnership.
- **SWOT Analysis**: Agile team; limited scale.
- **Competitive Threat Level**: Medium
**STRATEGIZE**
- **Competitive Strategy**: Differentiate on service.
- **Actionable Recommendations**: Improve distribution.
## COMPETITIVE COMPARISON MATRIX
| Feature | Alpha | Beta |
|---|---|---|
## MARKET OPPORTUNITIES
1. Opportunity one
2. Opportunity two
## STRATEGIC RECOMMENDATIONS
1. Act now
"""


class FakeSearch:
    response = {}
    last_instance = None

    def __init__(self, params):
        self.params = params
        self.timeout = None
        FakeSearch.last_instance = self

    def get_dict(self):
        return self.response


class SearchToolTests(unittest.TestCase):
    def test_official_website_can_be_inferred_from_repeated_branded_results(self):
        results = {
            "organic_results": [
                {"title": "Investor Relations | Porsche AG", "source": "Porsche AG", "link": "https://investorrelations.porsche.com/"},
                {"title": "Reports | Porsche AG", "source": "Porsche AG", "link": "https://investorrelations.porsche.com/reports"},
                {"title": "Porsche review", "source": "Unknown", "link": "https://unknown.example/porsche"},
            ]
        }
        self.assertEqual(
            infer_official_website(results, "Porsche"),
            "https://investorrelations.porsche.com/",
        )

    @patch("tools.GoogleSearch", FakeSearch)
    def test_serpapi_errors_are_not_treated_as_empty_results(self):
        FakeSearch.response = {"error": "Invalid API key"}
        with self.assertRaisesRegex(SearchAPIError, "Invalid API key"):
            execute_serpapi_search({"q": "test", "api_key": "bad"})

    @patch("tools.GoogleSearch", FakeSearch)
    def test_serpapi_timeout_and_result_processing(self):
        FakeSearch.response = {
            "organic_results": [
                {
                    "title": "Alpha",
                    "link": "https://www.reuters.com/technology/alpha",
                    "snippet": "A competitor",
                }
            ]
        }
        payload = json.loads(CompetitorSearchTool(api_key="test", max_results=1)._run("Acme competitors"))
        self.assertEqual(payload["organic_results"][0]["title"], "Alpha")
        self.assertEqual(payload["organic_results"][0]["trust_classification"], "recognized business press")
        self.assertGreater(FakeSearch.last_instance.timeout, 0)

    @patch("tools.GoogleSearch", FakeSearch)
    def test_unrecognized_search_results_are_excluded(self):
        FakeSearch.response = {
            "organic_results": [
                {"title": "Unknown blog", "link": "https://unknown.example/post", "snippet": "Claim"},
                {"title": "Official filing", "link": "https://www.sec.gov/Archives/test", "snippet": "10-K"},
            ]
        }
        payload = json.loads(CompetitorSearchTool(api_key="test")._run("Acme filing"))
        self.assertEqual([item["title"] for item in payload["organic_results"]], ["Official filing"])
        self.assertEqual(payload["untrusted_results_excluded"], 1)

    @patch("tools.execute_serpapi_search")
    def test_official_website_search_restricts_results_to_verified_domain(self, search):
        search.side_effect = [
            {"knowledge_graph": {"website": "https://www.porsche.com"}},
            {
                "organic_results": [
                    {"title": "Annual report", "link": "https://investorrelations.porsche.com/report"},
                    {"title": "Untrusted", "link": "https://unknown.example/porsche"},
                ]
            },
        ]
        payload = json.loads(OfficialWebsiteSearchTool(api_key="test")._run("Porsche | annual report"))
        self.assertEqual(payload["official_domain"], "porsche.com")
        self.assertEqual([item["title"] for item in payload["organic_results"]], ["Annual report"])

    @patch("tools.CompetitorSearchTool._run")
    def test_coverage_search_removes_trusted_but_irrelevant_results(self, search):
        search.return_value = json.dumps({
            "organic_results": [
                {"title": "Porsche market outlook", "link": "https://www.reuters.com/porsche", "snippet": "Porsche"},
                {"title": "Unrelated filing", "link": "https://www.sec.gov/unrelated", "snippet": "Other issuer"},
            ]
        })
        payload = json.loads(CoverageReportSearchTool(api_key="test")._run("Porsche initiating coverage"))
        self.assertEqual([item["title"] for item in payload["organic_results"]], ["Porsche market outlook"])
        self.assertEqual(payload["irrelevant_results_excluded"], 1)

    def test_data_processor_accepts_list_and_competitors_keyword(self):
        tool = DataProcessorTool()
        from_list = json.loads(tool._run('[{"name":"BMW"}]'))
        from_keyword = json.loads(tool._run(competitors=[{"name": "Audi"}]))
        self.assertEqual(from_list["competitors"][0]["name"], "BMW")
        self.assertEqual(from_keyword["competitors"][0]["name"], "Audi")


class ReportUtilityTests(unittest.TestCase):
    def test_report_sections_preserve_summary_and_findings(self):
        sections = format_report_for_display(SAMPLE_REPORT)
        self.assertIn("crowded market", sections["overview"])
        self.assertIn("Finding one", sections["overview"])
        self.assertIn("Alpha", sections["detailed_analysis"])
        self.assertIn("Act now", sections["recommendations"])

    def test_metrics_only_count_competitor_headings(self):
        metrics = extract_key_metrics(SAMPLE_REPORT)
        self.assertEqual(metrics["num_competitors"], 2)
        self.assertEqual(len(metrics["key_findings"]), 2)
        self.assertEqual(metrics["opportunities"], 2)
        self.assertEqual(metrics["threat_level"], "High")

    def test_pdf_generation_handles_long_numbered_lists(self):
        pdf = PDFReportGenerator("A & B", "R&D").generate_pdf("## List\n10. Tenth item")
        self.assertTrue(pdf.getvalue().startswith(b"%PDF"))

    def test_incomplete_report_is_rejected_before_saving(self):
        self.assertTrue(competitor_report_issues("The report above is complete.", 3))
        self.assertEqual(competitor_report_issues(SAMPLE_REPORT + "\n## TRUSTED SOURCE REGISTER\n" + ("Evidence. " * 200), 2), [])

    def test_report_without_deep_dive_framework_is_rejected(self):
        stripped = SAMPLE_REPORT.replace("- **Talent & Culture**: Specialist team.\n", "")
        issues = competitor_report_issues(
            stripped + "\n## TRUSTED SOURCE REGISTER\n" + ("Evidence. " * 200),
            2,
        )
        self.assertTrue(any("incomplete competitor deep-dives" in issue for issue in issues))

    def test_unquantified_deep_dive_fields_are_rejected(self):
        stripped = SAMPLE_REPORT.replace(
            "$99 per month as of 2025. (Source: Official pricing)",
            "Premium pricing",
        )
        issues = competitor_report_issues(
            stripped + "\n## TRUSTED SOURCE REGISTER\n" + ("Evidence. " * 200),
            2,
        )
        self.assertTrue(any("quantitative fields require" in issue for issue in issues))

    def test_unquantified_fields_are_disclosed_without_inventing_numbers(self):
        stripped = SAMPLE_REPORT.replace(
            "$99 per month as of 2025. (Source: Official pricing)",
            "Premium pricing",
        )
        repaired = enforce_quantification_disclosures(stripped)
        self.assertIn("Unknown / not verified. Qualitative context: Premium pricing", repaired)
        issues = competitor_report_issues(
            repaired + "\n## TRUSTED SOURCE REGISTER\n" + ("Evidence. " * 200),
            2,
        )
        self.assertFalse(any("quantitative fields require" in issue for issue in issues))

    def test_uncited_numbers_are_removed(self):
        uncited = SAMPLE_REPORT.replace(
            "$99 per month as of 2025. (Source: Official pricing)",
            "$129 per month as of 2025",
        )
        repaired = enforce_quantification_disclosures(uncited)
        self.assertNotIn("$129", repaired)
        self.assertIn("supplied magnitude had no inline trusted-source attribution", repaired)


if __name__ == "__main__":
    unittest.main()
