import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from judge import JudgeUnavailable, judge_once, main, score_row, summarize

ROW = {
    "case": "breach", "group": "shared", "workflow": "react",
    "question": "Why was the fund flagged?",
    "expected_answer": "165% versus 160%, breach", "expected_evidence": "EXPOSURE, POLICY",
    "answer": "Gross exposure is 165%, exceeding the 160% limit.",
    "cited_evidence": json.dumps({"EXPOSURE": "Gross exposure 165%.", "POLICY": "Limit 160%."}),
    "error": "", "answer_correct": "human-note", "citations_support_answer": "",
}
VERDICT = {
    "answer_correct": 1, "citations_support_answer": 1,
    "answer_reason": "The numbers and status match.", "citation_reason": "Both claims are supported.",
}


def response(verdict=VERDICT, finish="stop"):
    return io.StringIO(json.dumps({
        "choices": [{"finish_reason": finish, "message": {"content": json.dumps(verdict)}}],
        "usage": {"total_tokens": 100},
    }))


class JudgeTests(unittest.TestCase):
    @patch("judge.urlopen")
    def test_vllm_contract_and_blinded_prompt(self, urlopen):
        urlopen.return_value = response()
        verdict, tokens = judge_once(ROW, "http://127.0.0.1:8001/v1", "qwen3-8b", 30)
        self.assertEqual(verdict, VERDICT)
        self.assertEqual(tokens, 100)
        payload = json.loads(urlopen.call_args.args[0].data)
        record = json.loads(payload["messages"][1]["content"])
        self.assertNotIn("workflow", record)
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["response_format"]["type"], "json_schema")

    @patch("judge.urlopen")
    def test_invalid_or_truncated_response_cannot_be_scored(self, urlopen):
        for invalid in ({**VERDICT, "answer_correct": True}, {**VERDICT, "answer_correct": 2}):
            urlopen.return_value = response(invalid)
            with self.assertRaises(JudgeUnavailable):
                judge_once(ROW, "http://localhost/v1", "qwen3-8b", 30)
        urlopen.return_value = response(finish="length")
        with self.assertRaises(JudgeUnavailable):
            judge_once(ROW, "http://localhost/v1", "qwen3-8b", 30)

    @patch("judge.urlopen")
    def test_repeated_scores_and_failed_generation(self, urlopen):
        urlopen.side_effect = [response(), response({**VERDICT, "answer_correct": 0})]
        scored = score_row(ROW, "http://localhost/v1", "qwen3-8b", 2, 30)
        self.assertEqual(scored["judge_answer_correct"], 0.5)
        self.assertEqual(scored["judge_total_tokens"], 200)
        self.assertEqual(scored["answer_correct"], "human-note")
        failed = score_row({**ROW, "error": "Gemini timeout"}, "", "", 2, 30)
        self.assertEqual(failed["judge_status"], "generation_error")
        self.assertEqual(failed["judge_answer_correct"], 0)
        self.assertEqual(urlopen.call_count, 2)

    @patch("judge.urlopen", side_effect=TimeoutError)
    def test_judge_outage_leaves_scores_blank(self, urlopen):
        scored = score_row(ROW, "http://localhost/v1", "qwen3-8b", 1, 30)
        self.assertEqual(scored["judge_status"], "judge_error")
        self.assertEqual(scored["judge_answer_correct"], "")

    @patch("judge.urlopen")
    def test_csv_cli_keeps_manual_fields_and_refuses_overwrite(self, urlopen):
        urlopen.side_effect = lambda *args, **kwargs: response()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.csv"
            target = Path(folder) / "output.csv"
            with source.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(ROW))
                writer.writeheader()
                writer.writerow(ROW)
            args = ["judge.py", str(source), "--output", str(target)]
            with patch.object(sys, "argv", args), patch("sys.stdout", new_callable=io.StringIO):
                self.assertEqual(main(), 0)
            with target.open(encoding="utf-8-sig", newline="") as file:
                scored = next(csv.DictReader(file))
            self.assertEqual(scored["answer_correct"], "human-note")
            self.assertEqual(scored["judge_answer_correct"], "1.0")
            with patch.object(sys, "argv", args), patch("sys.stderr", new_callable=io.StringIO):
                with self.assertRaises(SystemExit):
                    main()

    @patch("judge.urlopen")
    def test_invalid_verdict_preserves_candidate_and_validation_reason(self, urlopen):
        urlopen.return_value = response({**VERDICT, "citation_reason": " "})
        scored = score_row(ROW, "http://localhost/v1", "qwen3-4b", 1, 30)
        self.assertEqual(scored["judge_status"], "judge_error")
        self.assertEqual(scored["judge_citations_support_answer"], "")
        self.assertIn("citation_reason", scored["judge_error"])
        self.assertEqual(json.loads(scored["judge_rejected_response"])["citation_reason"], " ")

    @patch("judge.urlopen")
    def test_empty_citations_cannot_earn_support(self, urlopen):
        urlopen.return_value = response()
        scored = score_row({**ROW, "cited_evidence": "{}"}, "http://localhost/v1", "qwen3-4b", 1, 30)
        self.assertEqual(scored["judge_answer_correct"], 1)
        self.assertEqual(scored["judge_citations_support_answer"], 0)

    def test_summary_reports_scored_subset_and_both_failure_types(self):
        rows = [
            {**ROW, "judge_status": "ok", "judge_answer_correct": 1, "judge_citations_support_answer": 0},
            {**ROW, "judge_status": "generation_error"},
            {**ROW, "judge_status": "judge_error"},
        ]
        with patch("sys.stdout", new_callable=io.StringIO) as output:
            summarize(rows)
        summary = output.getvalue()
        self.assertIn("1 generation failures, 1 judge failures", summary)
        self.assertIn("All-attempt scores unavailable", summary)
        self.assertIn("1 scored answers: answer 100.0%, citations 0.0%", summary)
