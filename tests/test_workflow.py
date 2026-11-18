import io
import json
import unittest
from unittest.mock import patch

from gemini import ExplanationUnavailable, RequestPacer, explain
from risk import investigate, load_demo, shock_position


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("gemini.APP_PACER", RequestPacer(0)))

    def test_breach_and_attribution(self):
        report = investigate(*load_demo())
        self.assertEqual((report["previous_gross"], report["current_gross"]), (145, 165))
        self.assertTrue(report["breach"])
        self.assertEqual([row["change_pp"] for row in report["contributions"]], [15, 5, 0])
        self.assertEqual(sum(row["change_pp"] for row in report["contributions"]), 20)

    def test_limit_equality_is_within_limit(self):
        navs, snapshots, policy = load_demo()
        self.assertFalse(investigate(navs, snapshots, {**policy, "gross_limit_pct": 165})["breach"])

    def test_scenario_revalues_nav_and_exposure(self):
        navs, snapshots, _ = load_demo()
        date = max(navs)
        result = shock_position(snapshots[date], navs[date], "ALPHA", -5)
        self.assertEqual(result["pnl"], -45000)
        self.assertEqual(result["nav"], 955000)
        self.assertAlmostEqual(result["gross_pct"], 168.0628272251)

    @patch("gemini.urlopen")
    def test_gemini_response_with_citation(self, urlopen):
        answer = json.dumps({"answer": "The reason is unknown.", "sources": ["NOTES"]})
        body = {"candidates": [{"content": {"parts": [{"text": answer}]}}]}
        urlopen.return_value = io.StringIO(json.dumps(body))
        self.assertEqual(
            explain("Why were the trades made?", {"NOTES": "No rationale available."}, "demo"),
            ("The reason is unknown.", ["NOTES"]),
        )
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 20)

    @patch("gemini.urlopen")
    def test_unknown_citation_is_rejected(self, urlopen):
        answer = json.dumps({"answer": "The answer.", "sources": ["MADE-UP"]})
        body = {"candidates": [{"content": {"parts": [{"text": answer}]}}]}
        urlopen.return_value = io.StringIO(json.dumps(body))
        with self.assertRaises(ExplanationUnavailable):
            explain("What happened?", {"NOTES": "A note."}, "demo")

    @patch("gemini.urlopen", side_effect=TimeoutError)
    def test_timeout_and_missing_key(self, urlopen):
        with self.assertRaises(ExplanationUnavailable):
            explain("What happened?", {}, "demo")
        with self.assertRaises(ExplanationUnavailable):
            explain("What happened?", {}, None)
        urlopen.assert_called_once()

    @patch("gemini.sleep")
    @patch("gemini.urlopen")
    def test_rate_limit_retry_and_http_error_diagnostic(self, urlopen, sleep):
        import io
        import json
        from urllib.error import HTTPError
        from gemini import GeminiSession
        body = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"totalTokenCount": 10}}
        rate_limit = HTTPError("https://example.com", 429, "rate limited", {}, None)
        urlopen.side_effect = [rate_limit, io.StringIO(json.dumps(body))]
        session = GeminiSession("mock-key")
        session.generate({})
        self.assertEqual(session.calls, 2)
        sleep.assert_called_once_with(30)
        urlopen.side_effect = HTTPError("https://example.com", 403, "forbidden", {}, None)
        with self.assertRaisesRegex(ExplanationUnavailable, "HTTP 403"):
            session.generate({})
