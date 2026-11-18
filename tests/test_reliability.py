import io
import json
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from agent import run_tool
from gemini import ExplanationUnavailable, GeminiSession, RequestPacer, parse_answer
from risk import investigate, load_demo


def provider_response():
    return io.StringIO(json.dumps({
        "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
        "usageMetadata": {"totalTokenCount": 10},
    }))


class ReliabilityTests(unittest.TestCase):
    def test_ranking_source_includes_all_positions_and_is_returned_by_tool(self):
        navs, snapshots, policy = load_demo()
        report = investigate(navs, snapshots, policy)
        result = run_tool("get_attribution", {}, report, snapshots[max(snapshots)], navs[max(navs)])
        ranking = result["ATTRIBUTION-RANKING"]
        self.assertIn("ALPHA +15 pp; BETA +5 pp; GAMMA +0 pp", ranking)
        self.assertIn("Complete position comparison", ranking)

    def test_ranking_rejection_retains_answer_and_evidence(self):
        evidence = investigate(*load_demo())["evidence"]
        content = {"parts": [{"text": json.dumps({
            "answer": "ALPHA is the largest contributor at +15 pp.",
            "sources": ["POSITION-ALPHA"],
        })}]}
        with self.assertRaises(ExplanationUnavailable) as caught:
            parse_answer(content, evidence)
        self.assertIn("ALPHA", caught.exception.raw_response)
        self.assertEqual(caught.exception.evidence, evidence)
        content["parts"][0]["text"] = json.dumps({
            "answer": "ALPHA is the largest contributor at +15 pp.",
            "sources": ["ATTRIBUTION-RANKING"],
        })
        self.assertEqual(parse_answer(content, evidence)[1], ["ATTRIBUTION-RANKING"])

    def test_missing_data_can_be_cited(self):
        evidence = investigate(*load_demo())["evidence"]
        content = {"parts": [{"text": json.dumps({
            "answer": "DELTA has no position data in this demo.",
            "sources": ["PORTFOLIO-SCOPE"],
        })}]}
        self.assertEqual(parse_answer(content, evidence)[1], ["PORTFOLIO-SCOPE"])

    @patch("gemini.urlopen")
    def test_pacing_is_shared_between_sessions(self, urlopen):
        clock = [0.0]
        def sleep(seconds):
            clock[0] += seconds
        urlopen.side_effect = lambda *args, **kwargs: provider_response()
        with patch("gemini.perf_counter", side_effect=lambda: clock[0]), patch("gemini.sleep", side_effect=sleep):
            pacer = RequestPacer(6)
            first = GeminiSession("key", pacer=pacer)
            first.generate({})
            second = GeminiSession("key", pacer=pacer)
            second.generate({})
            self.assertEqual(clock[0], 6)
            self.assertEqual(second.metrics()["wait_ms"], 6000)

    @patch("gemini.sleep")
    @patch("gemini.urlopen")
    def test_retry_respects_provider_delay_and_redacts_daily_quota_details(self, urlopen, sleep):
        body = {"error": {"details": [{"retryDelay": "45s"}]}}
        urlopen.side_effect = [
            HTTPError("https://example.com", 429, "", {}, io.BytesIO(json.dumps(body).encode())),
            provider_response(),
        ]
        GeminiSession("secret", pacer=RequestPacer(0)).generate({})
        sleep.assert_called_once_with(45)
        daily = b'{"error":{"message":"secret exceeded RequestsPerDay"}}'
        urlopen.side_effect = HTTPError("https://example.com", 429, "", {}, io.BytesIO(daily))
        with self.assertRaises(ExplanationUnavailable) as caught:
            GeminiSession("secret", pacer=RequestPacer(0)).generate({})
        self.assertNotIn("secret", caught.exception.http_details)
        self.assertIn("[REDACTED]", caught.exception.http_details)
        self.assertEqual(sleep.call_count, 1)

    @patch("gemini.sleep")
    @patch("gemini.urlopen")
    def test_retry_after_header_is_used_even_when_response_body_is_not_json(self, urlopen, sleep):
        urlopen.side_effect = [
            HTTPError("https://example.com", 503, "unavailable", {"Retry-After": "25"},
                      io.BytesIO(b"Temporary outage")),
            provider_response(),
        ]
        GeminiSession("secret", pacer=RequestPacer(0)).generate({})
        sleep.assert_called_once_with(25)
