import unittest

from trusted_sources import (
    build_trusted_source_context,
    classify_trusted_url,
    extract_report_text,
    trusted_organic_results,
)


class TrustedSourcePolicyTests(unittest.TestCase):
    def test_recognized_publishers_and_regulators_are_trusted(self):
        self.assertEqual(
            classify_trusted_url("https://www.jpmorgan.com/insights/research/example"),
            "investment-bank research",
        )
        self.assertEqual(
            classify_trusted_url("https://www.sec.gov/Archives/edgar/data/example"),
            "regulatory filing",
        )

    def test_unknown_domain_is_rejected_unless_explicitly_approved(self):
        url = "https://research.example/report"
        self.assertIsNone(classify_trusted_url(url))
        self.assertEqual(
            classify_trusted_url(url, user_domains=["research.example"]),
            "user-approved report or official source",
        )

    def test_knowledge_graph_official_domain_is_dynamically_trusted(self):
        results = {
            "knowledge_graph": {"website": "https://www.acme.example"},
            "organic_results": [
                {"title": "Official", "link": "https://investors.acme.example/report"},
                {"title": "Blog", "link": "https://random.example/acme"},
            ],
        }
        accepted, rejected = trusted_organic_results(results)
        self.assertEqual([item["title"] for item in accepted], ["Official"])
        self.assertEqual(rejected, ["random.example"])

    def test_uploaded_text_and_html_are_extracted(self):
        self.assertEqual(extract_report_text("note.txt", b"Trusted report"), "Trusted report")
        html = b"<html><body><nav>Menu</nav><h1>Outlook</h1><p>Demand rises.</p></body></html>"
        extracted = extract_report_text("report.html", html, "text/html")
        self.assertIn("Outlook", extracted)
        self.assertNotIn("Menu", extracted)

    def test_private_urls_are_rejected_before_download(self):
        context, domains, errors = build_trusted_source_context(["http://127.0.0.1/report"], [])
        self.assertEqual(context, "")
        self.assertEqual(domains, [])
        self.assertTrue(errors)
        self.assertIn("Private or local", errors[0])


if __name__ == "__main__":
    unittest.main()
