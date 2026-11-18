import json

from gemini import ANSWER_INSTRUCTIONS, ExplanationUnavailable, parse_answer, response_parts
from risk import scenario_evidence

MAX_TURNS = 4
MAX_TOOL_CALLS = 6
TOOLS = [
    {
        "name": "get_exposure",
        "description": "Get the latest gross exposure, prior exposure and applicable limit.",
    },
    {
        "name": "get_attribution",
        "description": "Get position contributions to the change in gross exposure, with source rows.",
    },
    {
        "name": "run_scenario",
        "description": "Revalue one position for a signed hypothetical price change on the latest snapshot.",
        "parameters": {
            "type": "OBJECT",
            "properties": {"symbol": {"type": "STRING"}, "shock_pct": {"type": "NUMBER"}},
            "required": ["symbol", "shock_pct"],
        },
    },
]


def run_tool(name, args, report, positions, nav):
    evidence = report["evidence"]
    if name == "get_exposure" and args == {}:
        return {key: evidence[key] for key in ("EXPOSURE", "POLICY-2.1", "PORTFOLIO-SCOPE")}
    if name == "get_attribution" and args == {}:
        return {
            "ATTRIBUTION-RANKING": evidence["ATTRIBUTION-RANKING"],
            **{row["source"]: evidence[row["source"]] for row in report["contributions"]},
        }
    if name != "run_scenario" or not isinstance(args, dict) or args.keys() != {"symbol", "shock_pct"}:
        raise ValueError("Unknown tool or invalid arguments.")
    symbol, shock = args["symbol"], args["shock_pct"]
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("Provide a nonempty position symbol.")
    symbol = symbol.strip()
    if symbol not in positions and symbol.upper() in positions:
        symbol = symbol.upper()
    if type(shock) not in (int, float) or not -float("inf") < shock < float("inf"):
        raise ValueError("Provide a numeric price move between -100% and +100%.")
    if not -100 <= shock <= 100:
        raise ValueError(
            f"Scenario for {symbol.upper()} at {shock:+g}% is unavailable. "
            "The supported price move is between -100% and +100%."
        )
    if symbol not in positions:
        raise ValueError(
            f"Scenario for {symbol.upper()} at {shock:+g}% is unavailable. "
            f"{symbol} is absent from the latest snapshot ({report['current_date']})."
        )
    source = f"SHOCK-{symbol}-{float(shock)!r}"
    try:
        return {source: scenario_evidence(positions, nav, symbol, shock, report["limit"])}
    except ValueError as error:
        raise ValueError(f"Scenario for {symbol} at {shock:+g}% is unavailable. {error}") from None


def agent_workflow(question, report, positions, nav, session):
    evidence = {"NOTES": report["evidence"]["NOTES"]}
    calls = []

    def reject_turn(message, content):
        error = ExplanationUnavailable(
            message, raw_response=json.dumps(response_parts(content)), evidence=evidence
        )
        error.tool_calls = list(calls)
        return error

    history = [{"role": "user", "parts": [{"text": (
        ANSWER_INSTRUCTIONS + " Use the read-only tools to collect the facts you need. "
        "A fund flag refers to a policy breach, which is distinct from the manager's trade rationale. "
        "Use get_exposure to explain a flag, get_attribution for contributors, and run_scenario "
        "for a hypothetical shock. Never assume a tool succeeded. Use the latest snapshot.\n"
        + json.dumps({"question": question, "evidence": evidence})
    )}]}]
    for turn in range(MAX_TURNS + 1):
        final_turn = turn == MAX_TURNS or len(calls) >= MAX_TOOL_CALLS
        calling_config = {"mode": "NONE" if final_turn else "AUTO"}
        if turn == 0:
            calling_config = {"mode": "ANY", "allowedFunctionNames": ["get_exposure"]}
        payload = {
            "contents": history,
            "tools": [{"functionDeclarations": TOOLS}],
            "toolConfig": {"functionCallingConfig": calling_config},
        }
        try:
            content = session.generate(payload)
        except ExplanationUnavailable as error:
            error.evidence = dict(evidence)
            error.tool_calls = list(calls)
            raise
        parts = content.get("parts", [])
        if any(not isinstance(part, dict) for part in parts):
            raise reject_turn("Gemini returned an invalid tool turn.", content)
        requested = [part["functionCall"] for part in parts if "functionCall" in part]
        if any(not isinstance(call, dict) or not isinstance(call.get("name"), str) for call in requested):
            raise reject_turn("Gemini returned an invalid tool call.", content)
        if turn == 0 and (
            len(requested) != 1 or requested[0]["name"] != "get_exposure"
            or requested[0].get("args", {}) != {}
        ):
            raise reject_turn("Gemini skipped the required exposure check.", content)
        if not requested:
            try:
                answer, sources = parse_answer(content, evidence, question=question)
            except ExplanationUnavailable as error:
                error.tool_calls = list(calls)
                raise
            return {"answer": answer, "sources": sources, "evidence": evidence, "tool_calls": calls}
        if final_turn or len(calls) + len(requested) > MAX_TOOL_CALLS:
            raise reject_turn("Gemini exceeded the investigation tool budget.", content)

        responses = []
        for call in requested:
            name, args = call.get("name", ""), call.get("args", {})
            try:
                facts = run_tool(name, args, report, positions, nav)
            except ValueError as error:
                source = f"TOOL-ERROR-{len(calls) + 1}"
                evidence[source] = str(error)
                response = {"error": str(error), "source": source}
            else:
                evidence.update(facts)
                response = {"evidence": facts}
            part = {"name": name, "response": response}
            if "id" in call:
                part["id"] = call["id"]
            calls.append({
                "tool": name, "arguments": args, "status": "ok" if "evidence" in response else "error",
                "response": response, **({"id": call["id"]} if "id" in call else {}),
            })
            responses.append({"functionResponse": part})
        history.extend([content, {"role": "user", "parts": responses}])
    raise ExplanationUnavailable("Gemini did not finish the investigation.")
