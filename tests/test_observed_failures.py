import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from agent import agent_workflow
from gemini import ExplanationUnavailable, GeminiSession, RequestPacer, fixed_workflow
from judge import score_row
from risk import fixed_evidence_for, investigate, load_demo


def answer(text, *sources):
    return {"parts": [{"text": json.dumps({"answer": text, "sources": sources})}]}


class ObservedFailures(unittest.TestCase):
    def setUp(self):
        navs, snapshots, policy = load_demo()
        self.report = investigate(navs, snapshots, policy)
        date = self.report["current_date"]
        self.positions, self.nav = snapshots[date], navs[date]
        self.evidence = fixed_evidence_for(self.report, self.positions, self.nav, "ALPHA")

    def test_rank_question_requires_comparison_even_without_superlative_in_answer(self):
        session = Mock(generate=Mock(return_value=answer("ALPHA added 15 pp.", "POSITION-ALPHA")))
        with self.assertRaisesRegex(ExplanationUnavailable, "Ranking"):
            fixed_workflow("Identify the largest driver.", self.evidence, session)

    def test_wrong_scenario_is_rejected_and_grounded_gap_is_accepted(self):
        question = "What is the P&L and gross exposure if BETA rises 10%?"
        session = Mock(generate=Mock(return_value=answer("Exposure is 168.06%.", "SHOCK")))
        with self.assertRaisesRegex(ExplanationUnavailable, "scenario"):
            fixed_workflow(question, self.evidence, session)
        session.generate.return_value = answer("No BETA +10% scenario is available.", "SCENARIO-COVERAGE")
        self.assertEqual(fixed_workflow(question, self.evidence, session)["sources"], ["SCENARIO-COVERAGE"])
        session.generate.return_value = answer(
            "No BETA scenario is available. Gross exposure would reach 168.06%.", "SCENARIO-COVERAGE"
        )
        with self.assertRaisesRegex(ExplanationUnavailable, "scenario"):
            fixed_workflow(question, self.evidence, session)

    def test_react_rejects_wrong_first_call_with_context(self):
        content = {"parts": [{"functionCall": {"name": "get_attribution"}}]}
        session = Mock(generate=Mock(return_value=content))
        with self.assertRaisesRegex(ExplanationUnavailable, "required exposure check") as caught:
            agent_workflow("Why was the fund flagged?", self.report, self.positions, self.nav, session)
        self.assertIn("get_attribution", caught.exception.raw_response)
        self.assertIn("NOTES", caught.exception.evidence)
        session.generate.assert_called_once()

    def test_missing_position_and_unavailable_nav_produce_citable_tool_errors(self):
        exposure = {"parts": [{"functionCall": {"name": "get_exposure"}}]}
        for symbol, nav, reason in (("DELTA", self.nav, "absent"), ("ALPHA", 1, "NAV must be positive")):
            with self.subTest(symbol=symbol):
                tool = {"parts": [{"functionCall": {
                    "name": "run_scenario", "args": {"symbol": symbol, "shock_pct": -5},
                }}]}
                final = answer(f"No {symbol} scenario is available.", "TOOL-ERROR-2")
                session = Mock(generate=Mock(side_effect=[exposure, tool, final]))
                result = agent_workflow(f"What if {symbol} fell 5%?", self.report, self.positions, nav, session)
                self.assertIn(reason, result["evidence"]["TOOL-ERROR-2"])
                self.assertEqual(result["tool_calls"][1]["response"]["source"], "TOOL-ERROR-2")

    def test_legacy_judge_cannot_award_full_credit_for_single_position(self):
        row = {
            "question": "Identify the largest driver.", "answer": "ALPHA added 15 pp.",
            "cited_evidence": '{"POSITION-ALPHA":"+15 pp"}', "error": "",
            "answer_correct": "1", "citations_support_answer": "human-review",
        }
        with patch("judge.judge_once", return_value=({"answer_correct": 1, "citations_support_answer": 1}, 5)):
            scored = score_row(row, "", "qwen3-4b", 1, 30)
            fully_cited = {**row, "cited_evidence": json.dumps({
                key: self.evidence[key] for key in ("POSITION-ALPHA", "POSITION-BETA", "POSITION-GAMMA")
            })}
            fully_scored = score_row(fully_cited, "", "qwen3-4b", 1, 30)
        self.assertEqual(scored["judge_citations_support_answer"], 0)
        self.assertEqual(scored["judge_answer_correct"], 1)
        self.assertEqual(scored["citations_support_answer"], "human-review")
        self.assertEqual(fully_scored["judge_citations_support_answer"], 1)

    @patch("gemini.sleep")
    @patch("gemini.urlopen")
    def test_recovered_http_error_is_retained_and_all_attempts_use_pacer(self, urlopen, sleep):
        error = HTTPError("https://example.com", 429, "quota", {"Retry-After": "40"},
                          io.BytesIO(b'{"error":{"message":"minute quota for secret-key"}}'))
        exposure = {"parts": [{"functionCall": {"name": "get_exposure"}}]}
        replies = iter([exposure, error, answer("No rationale.", "NOTES")])
        clock = [0]
        starts = []

        def advance(seconds):
            clock[0] += seconds

        def request(*args, **kwargs):
            starts.append(clock[0])
            reply = next(replies)
            if isinstance(reply, HTTPError):
                raise reply
            return io.StringIO(json.dumps({"candidates": [{"content": reply}]}))

        sleep.side_effect = advance
        urlopen.side_effect = request
        with patch("gemini.perf_counter", side_effect=lambda: clock[0]):
            session = GeminiSession("secret-key", pacer=RequestPacer(6))
            agent_workflow("Why did the manager make the trades?", self.report, self.positions, self.nav, session)
        self.assertEqual(starts, [0, 6, 46])
        self.assertEqual(session.api_errors[0]["status"], 429)
        self.assertIn("minute quota", session.api_errors[0]["body"])
        self.assertNotIn("secret-key", str(session.api_errors))

    def test_unavailable_scenario_cannot_reuse_shock_as_result(self):
        session = Mock(generate=Mock(return_value=answer(
            "No BETA scenario is available. Gross exposure would reach 10%.", "SCENARIO-COVERAGE"
        )))
        with self.assertRaisesRegex(ExplanationUnavailable, "scenario"):
            fixed_workflow("What is gross exposure if BETA rises 10%?", self.evidence, session)

    def test_lowercase_symbol_still_requires_exact_scenario(self):
        session = Mock(generate=Mock(return_value=answer("Exposure would be 168.06%.", "SHOCK")))
        with self.assertRaisesRegex(ExplanationUnavailable, "scenario"):
            fixed_workflow("What if beta rises 10%?", self.evidence, session)

    def test_explicit_absence_claim_requires_scope_or_matching_tool_error(self):
        session = Mock(generate=Mock(return_value=answer(
            "DELTA is not in the portfolio.", "SCENARIO-COVERAGE"
        )))
        with self.assertRaisesRegex(ExplanationUnavailable, "scenario"):
            fixed_workflow("What if DELTA fell 5%?", self.evidence, session)

    def test_judge_rejects_comparison_with_placeholder_passages(self):
        row = {
            "question": "Identify the largest driver.", "answer": "ALPHA added 15 pp.",
            "cited_evidence": json.dumps({f"POSITION-{s}": "unavailable" for s in self.positions}),
            "error": "",
        }
        with patch("judge.judge_once", return_value=({"answer_correct": 1, "citations_support_answer": 1}, 5)):
            scored = score_row(row, "", "qwen3-4b", 1, 30)
        self.assertEqual(scored["judge_citations_support_answer"], 0)

    @patch("gemini.urlopen")
    def test_candidate_rejection_retains_partial_reply_and_tool_trace(self, urlopen):
        exposure = {"parts": [{"functionCall": {"name": "get_exposure"}}]}
        urlopen.side_effect = [
            io.StringIO(json.dumps({"candidates": [{"content": exposure}]})),
            io.StringIO(json.dumps({"candidates": [{"finishReason": "MAX_TOKENS", "content": {"parts": [
                {"text": "partial reply"}, {"thought": True, "text": "private reasoning"},
                {"thoughtSignature": "private signature"},
            ]}}]})),
        ]
        with self.assertRaises(ExplanationUnavailable) as caught:
            agent_workflow("Why was the fund flagged?", self.report, self.positions, self.nav,
                           GeminiSession("secret", pacer=RequestPacer(0)))
        rejected = caught.exception
        self.assertIn("partial reply", rejected.raw_response)
        self.assertIn("MAX_TOKENS", rejected.raw_response)
        self.assertNotIn("private", rejected.raw_response)
        self.assertEqual(rejected.tool_calls[0]["tool"], "get_exposure")
        self.assertIn("EXPOSURE", rejected.evidence)

    @patch("gemini.urlopen")
    def test_standalone_explanations_share_request_pacing(self, urlopen):
        from gemini import explain

        starts, clock = [], [0]
        def request(*args, **kwargs):
            starts.append(clock[0])
            content = answer("The demo has no rationale.", "NOTES")
            return io.StringIO(json.dumps({"candidates": [{"content": content}]}))

        urlopen.side_effect = request
        with patch("gemini.APP_PACER", RequestPacer(6)), patch(
            "gemini.perf_counter", side_effect=lambda: clock[0]
        ), patch("gemini.sleep", side_effect=lambda delay: clock.__setitem__(0, clock[0] + delay)):
            for _ in range(2):
                explain("Why did the manager trade?", self.evidence, "secret")
        self.assertEqual(starts, [0, 6])

    def test_fixed_evidence_can_explain_coverage_without_claiming_position_absence(self):
        session = Mock(generate=Mock(return_value=answer(
            "The BETA +10% scenario is not in the fixed evidence.", "SCENARIO-COVERAGE"
        )))
        result = fixed_workflow("What if BETA rises 10%?", self.evidence, session)
        self.assertEqual(result["sources"], ["SCENARIO-COVERAGE"])

    def test_tool_error_gap_can_include_its_source_id_in_the_answer(self):
        session = Mock(generate=Mock(side_effect=[
            {"parts": [{"functionCall": {"name": "get_exposure"}}]},
            {"parts": [{"functionCall": {
                "name": "run_scenario", "args": {"symbol": "delta", "shock_pct": -5},
            }}]},
            answer("DELTA is absent from the latest snapshot (TOOL-ERROR-2).", "TOOL-ERROR-2"),
        ]))
        result = agent_workflow("What if delta fell 5%?", self.report, self.positions, self.nav, session)
        self.assertEqual(result["sources"], ["TOOL-ERROR-2"])
