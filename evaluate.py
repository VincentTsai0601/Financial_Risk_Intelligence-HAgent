"""Run a paired Gemini evaluation and create a human-reviewable CSV."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
from time import sleep

from agent import agent_workflow
from gemini import (
    DEFAULT_MODEL, DEFAULT_REQUEST_INTERVAL, ExplanationUnavailable, GeminiSession, RequestPacer, fixed_workflow,
)
from risk import fixed_evidence_for, investigate, load_demo

CASES = [
    ("breach", "Why was the fund flagged?", "165% versus 160%, breach", "EXPOSURE, POLICY-2.1"),
    ("change", "How much did gross exposure change since the previous snapshot?",
     "145% to 165%, +20 percentage points", "EXPOSURE"),
    ("attribution", "Which two positions contributed most to the increase?",
     "ALPHA +15 pp, BETA +5 pp", "ATTRIBUTION-RANKING"),
    ("short", "How much did the BETA short contribute to the change?",
     "BETA +5 pp", "POSITION-BETA"),
    ("unchanged", "Did GAMMA contribute to the exposure increase?",
     "GAMMA 0 pp", "POSITION-GAMMA"),
    ("scenario", "What is the P&L, NAV, and gross exposure if ALPHA falls 5%?",
     "-45,000 USD, 955,000 USD, 168.06%", "SHOCK or SHOCK-ALPHA--5.0"),
    ("combined", "Which position contributed most, and would the fund still breach if it fell 5%?",
     "ALPHA +15 pp, 168.06% vs 160%, still a breach",
     "ATTRIBUTION-RANKING, scenario evidence and/or policy"),
    ("paraphrase", "Identify the largest driver, then knock 5% off its price. Does exposure clear the cap?",
     "ALPHA +15 pp, 168.06% vs 160%, still a breach",
     "ATTRIBUTION-RANKING, scenario evidence and/or policy"),
    ("rationale", "Why did the manager make the trades?",
     "Insufficient evidence about trade rationale", "NOTES"),
    ("short_shock", "What is the P&L and gross exposure if BETA rises 10%?",
     "-55,000 USD, 180.42%", "SHOCK-BETA-10.0"),
    ("new_shock", "What is the P&L and gross exposure if GAMMA rises 10%?",
     "+20,000 USD, 163.73%", "SHOCK-GAMMA-10.0"),
    ("unknown_position", "What would happen if DELTA fell 5%?",
     "DELTA is absent from the latest snapshot, so its scenario cannot be computed. No invented values.",
     "PORTFOLIO-SCOPE or matching TOOL-ERROR"),
]
FIELDS = [
    "case", "group", "run", "workflow_version", "workflow", "model", "question", "expected_answer", "expected_evidence",
    "answer", "cited_evidence", "tool_calls", "api_calls", "total_tokens", "latency_ms",
    "error", "rejected_response", "available_evidence", "http_error_details", "api_errors",
    "wait_ms", "request_interval", "answer_correct", "citations_support_answer",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation.csv"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--delay", type=float, default=2, help="Seconds between workflow runs.")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL,
                        help="Minimum seconds between Gemini HTTP attempts, including tool turns and retries.")
    args = parser.parse_args()
    if not math.isfinite(args.request_interval) or args.request_interval < 0:
        parser.error("--request-interval must be finite and nonnegative.")
    if not math.isfinite(args.delay) or args.delay < 0:
        parser.error("--delay must be a nonnegative number.")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        parser.error("Set GEMINI_API_KEY before running the live evaluation.")
    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    pacer = RequestPacer(args.request_interval)
    navs, snapshots, policy = load_demo()
    report = investigate(navs, snapshots, policy)
    date = report["current_date"]
    positions, nav = snapshots[date], navs[date]
    fixed_evidence = fixed_evidence_for(report, positions, nav, "ALPHA")
    workflows = [
        ("fixed", lambda question, session: fixed_workflow(question, fixed_evidence, session)),
        ("react", lambda question, session: agent_workflow(question, report, positions, nav, session)),
    ]
    with args.output.open("x", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for repeat in range(1, args.repeats + 1):
            for index, (case, question, expected, sources) in enumerate(CASES):
                order = workflows if (repeat + index) % 2 else workflows[::-1]
                for name, workflow in order:
                    sleep(args.delay)
                    session = GeminiSession(key, model, pacer=pacer)
                    row = {
                        "case": case,
                        "group": "on_demand" if case in {"short_shock", "new_shock"} else "shared",
                        "workflow_version": "4",
                        "request_interval": args.request_interval,
                        "run": repeat, "workflow": name, "model": model,
                        "question": question, "expected_answer": expected, "expected_evidence": sources,
                    }
                    try:
                        result = workflow(question, session)
                    except ExplanationUnavailable as error:
                        row.update(
                            error=str(error),
                            rejected_response=error.raw_response,
                            available_evidence=json.dumps(error.evidence),
                            tool_calls=json.dumps(error.tool_calls),
                            http_error_details=error.http_details,
                        )
                    else:
                        row.update(
                            available_evidence=json.dumps(result["evidence"]),
                            answer=result["answer"],
                            cited_evidence=json.dumps({
                                source: result["evidence"][source] for source in result["sources"]
                            }),
                            tool_calls=json.dumps(result["tool_calls"]),
                        )
                    writer.writerow({**row, **session.metrics(), "api_errors": json.dumps(session.api_errors)})
                    file.flush()
                    print(f"{case} / {name} / run {repeat}", flush=True)
    print(f"Review answer accuracy and citation support in {args.output}.")


if __name__ == "__main__":
    main()
