import copy
import json
import unittest
from unittest.mock import Mock

from agent import agent_workflow, run_tool
from gemini import ExplanationUnavailable
from risk import investigate, load_demo


class AgentTests(unittest.TestCase):
    def setUp(self):
        navs, snapshots, policy = load_demo()
        self.report = investigate(navs, snapshots, policy)
        date = self.report["current_date"]
        self.positions, self.nav = snapshots[date], navs[date]

    def test_multi_step_investigation_preserves_model_parts_and_call_id(self):
        exposure_turn = {"role": "model", "parts": [{"functionCall": {"name": "get_exposure"}}]}
        tool_turn = {"role": "model", "parts": [{
            "thoughtSignature": "keep-this",
            "functionCall": {"name": "get_attribution", "id": "call-1"},
        }]}
        shock_turn = {"role": "model", "parts": [{
            "functionCall": {"name": "run_scenario", "args": {"symbol": "ALPHA", "shock_pct": -5}},
        }]}
        final = {"parts": [{"text": json.dumps({
            "answer": "ALPHA added 15 pp. After the shock, exposure remains above the limit.",
            "sources": ["ATTRIBUTION-RANKING", "SHOCK-ALPHA--5.0"],
        })}]}
        payloads = []
        responses = iter([exposure_turn, tool_turn, shock_turn, final])

        def generate(payload):
            payloads.append(copy.deepcopy(payload))
            return next(responses)

        result = agent_workflow(
            "What if the main contributor fell 5%?", self.report,
            self.positions, self.nav, Mock(generate=generate),
        )
        self.assertEqual(payloads[2]["contents"][3], tool_turn)
        response = payloads[2]["contents"][4]["parts"][0]["functionResponse"]
        self.assertEqual(response["id"], "call-1")
        self.assertIn("168.06%", result["evidence"][result["sources"][1]])
        self.assertEqual([call["status"] for call in result["tool_calls"]], ["ok", "ok", "ok"])

    def test_scenario_rejects_invalid_inputs(self):
        for args in ({"symbol": "DELTA", "shock_pct": -5}, {"symbol": "ALPHA", "shock_pct": True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                run_tool("run_scenario", args, self.report, self.positions, self.nav)

    def test_agent_cannot_cite_unfetched_evidence(self):
        final = {"parts": [{"text": '{"answer":"Breach","sources":["EXPOSURE"]}'}]}
        with self.assertRaises(ExplanationUnavailable):
            agent_workflow("Breach?", self.report, self.positions, self.nav,
                           Mock(generate=Mock(return_value=final)))

    def test_first_call_requests_exposure_before_answering(self):
        session = Mock(generate=Mock(return_value={"parts": [{
            "text": '{"answer":"No rationale","sources":["NOTES"]}'
        }]}))
        with self.assertRaisesRegex(ExplanationUnavailable, "required exposure check"):
            agent_workflow("Why was the fund flagged?", self.report, self.positions, self.nav, session)
        config = session.generate.call_args.args[0]["toolConfig"]["functionCallingConfig"]
        self.assertEqual(config, {"mode": "ANY", "allowedFunctionNames": ["get_exposure"]})
