import json
import math
import re
from decimal import Decimal
from threading import Lock
from time import perf_counter, sleep
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_REQUEST_INTERVAL = 6
RANKING = re.compile(
    r"\b(?:largest|biggest|smallest|top|leading|main|highest|lowest) "
    r"(?:\w+ ){0,2}(?:drivers?|contributors?|contributions?|positions?)\b|"
    r"\bcontribut\w* (?:the )?most\b|\bmost significant (?:driver|contributor)\b|"
    r"\brank(?:ed|ing)?\b", re.I
)
NAMED_SHOCK = re.compile(
    r"\b(?!(?:it|they|position|price|exposure|nav)\b)(?P<symbol>[A-Z][A-Z0-9.-]*) (?:price )?"
    r"(?P<direction>(?i:falls?|fell|drops?|declines?|rises?|rose|gains?|increases?)) "
    r"(?:by )?(?P<pct>\d+(?:\.\d+)?)\s*%", re.I
)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
ABSENCE_CLAIM = re.compile(
    r"\b(?:absent from|not held|not (?:listed )?in (?:the )?(?:portfolio|snapshot)|"
    r"no (?:current )?position(?: data)?|outside (?:the )?portfolio)\b",
    re.I,
)
# A quoted move can explain a gap, but it cannot serve as a computed metric.
SCENARIO_RESULT = re.compile(
    r"\b(?:P&L|NAV|(?:gross )?exposure|loss|profit)\s+"
    r"(?:(?:would|is|be|become|reach|of|at|falls?|rises?|to|equal)\s+)*"
    r"[:=]?\s*[-+−]?\$?\d",
    re.I,
)
ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answer": {"type": "STRING"},
        "sources": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["answer", "sources"],
}
ANSWER_INSTRUCTIONS = (
    "Use only the supplied evidence. Preserve values and units. Distinguish scenarios "
    "from observations. State when evidence is missing. Return a JSON object with "
    "answer (at most three sentences) and sources (the evidence IDs you used). "
    "For largest, most, top, or ranking claims cite ATTRIBUTION-RANKING. "
    "For an unavailable symbol cite PORTFOLIO-SCOPE. If a requested scenario is not "
    "provided and no scenario tool is available, state the specific gap and cite "
    "SCENARIO-COVERAGE. Cite the exact symbol and move for computed scenarios. In tool "
    "mode cite a relevant tool error if the calculation failed. Include citations for gaps."
)


class ExplanationUnavailable(Exception):
    def __init__(self, message, raw_response="", evidence=None, http_details=""):
        super().__init__(message)
        self.raw_response = raw_response
        self.evidence = dict(evidence or {})
        self.tool_calls = []
        self.http_details = http_details


class RequestPacer:
    """Share one pacer across sequential sessions in an evaluation run."""

    def __init__(self, interval):
        if not math.isfinite(interval) or interval < 0:
            raise ValueError("Request interval must be finite and nonnegative.")
        self.interval = interval
        self.next_allowed = 0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            delay = max(0, self.next_allowed - perf_counter())
            if delay:
                sleep(delay)
            self.next_allowed = perf_counter() + self.interval
            return delay


APP_PACER = RequestPacer(DEFAULT_REQUEST_INTERVAL)


def response_parts(content):
    """Keep user-visible response material in rejection diagnostics."""
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        return []
    return [
        {key: part[key] for key in ("text", "functionCall") if key in part}
        for part in content["parts"]
        if isinstance(part, dict) and not part.get("thought")
        and ("text" in part or "functionCall" in part)
    ]


class GeminiSession:
    def __init__(self, api_key, model=DEFAULT_MODEL, pacer=None):
        if not api_key:
            raise ExplanationUnavailable("Set GEMINI_API_KEY to enable Gemini explanations.")
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", model):
            raise ExplanationUnavailable("GEMINI_MODEL must be a Gemini model name.")
        self.api_key = api_key
        self.model = model
        self.calls = 0
        self.total_tokens = 0
        self.usage_complete = True
        self.started = perf_counter()
        self.pacer = pacer if pacer is not None else APP_PACER
        self.wait_seconds = 0
        self.api_errors = []

    def generate(self, payload):
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key},
        )
        for attempt in range(3):
            self.wait_seconds += self.pacer.wait()
            self.calls += 1
            try:
                with urlopen(request, timeout=20) as response:
                    raw = response.read()
                result = json.loads(raw)
            except HTTPError as error:
                self.usage_complete = False
                retry_after = (error.headers or {}).get("Retry-After", "")
                try:
                    with error:
                        body = error.read(8192) if error.fp is not None else b""
                except (OSError, ValueError):
                    body = b""
                details = body.decode("utf-8", errors="replace").replace(self.api_key, "[REDACTED]")
                self.api_errors.append({
                    "attempt": self.calls, "status": error.code,
                    "reason": str(error.reason).replace(self.api_key, "[REDACTED]"),
                    "retry_after": retry_after.replace(self.api_key, "[REDACTED]"), "body": details,
                })
                daily_quota = bool(re.search(r"requestsperday|per.day|daily", details, re.I))
                delay = 30 * (2 ** attempt) if error.code == 429 else 2 ** attempt
                try:
                    delay = max(delay, float(retry_after or 0))
                except ValueError:
                    pass
                try:
                    body = json.loads(details)
                    for item in body.get("error", {}).get("details", []):
                        if isinstance(item, dict) and "retryDelay" in item:
                            delay = max(delay, float(str(item["retryDelay"]).removesuffix("s")))
                except (ValueError, TypeError, AttributeError):
                    pass
                if (error.code in (429, 500, 502, 503, 504) and attempt < 2
                        and not daily_quota and math.isfinite(delay) and delay <= 60):
                    sleep(delay)
                    self.wait_seconds += delay
                    continue
                raise ExplanationUnavailable(
                    f"Gemini HTTP {error.code}. See http_error_details for quota or request details.",
                    http_details=details,
                ) from None
            except (URLError, TimeoutError, OSError) as error:
                self.usage_complete = False
                self.api_errors.append({
                    "attempt": self.calls, "type": type(error).__name__,
                    "reason": str(error).replace(self.api_key, "[REDACTED]"),
                })
                raise ExplanationUnavailable(
                    "Gemini connection failed or timed out after 20 seconds."
                ) from None
            except (ValueError, TypeError):
                self.usage_complete = False
                raise ExplanationUnavailable(
                    "Gemini returned invalid JSON.",
                    raw_response=str(raw).replace(self.api_key, "[REDACTED]"),
                ) from None
            try:
                token_count = result.get("usageMetadata", {}).get("totalTokenCount")
                if type(token_count) is int:
                    self.total_tokens += token_count
                else:
                    self.usage_complete = False
                candidate = result["candidates"][0]
                content = candidate["content"]
                if candidate.get("finishReason") not in (None, "STOP"):
                    raise ValueError("Candidate did not finish normally")
                if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
                    raise ValueError("Invalid model content")
                return content
            except (KeyError, TypeError, IndexError, AttributeError, ValueError):
                metadata = {}
                if isinstance(result, dict):
                    metadata["promptFeedback"] = result.get("promptFeedback")
                    candidates = result.get("candidates")
                    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
                        metadata["finishReason"] = candidates[0].get("finishReason")
                        metadata["parts"] = response_parts(candidates[0].get("content"))
                raise ExplanationUnavailable(
                    "Gemini returned no usable candidate. See the rejected response for provider details.",
                    raw_response=json.dumps(metadata).replace(self.api_key, "[REDACTED]"),
                ) from None

    def metrics(self):
        return {
            "api_calls": self.calls,
            "wait_ms": round(self.wait_seconds * 1000),
            "total_tokens": self.total_tokens if self.usage_complete else None,
            "latency_ms": round((perf_counter() - self.started) * 1000),
        }


def ranking_needs_comparison(text):
    return bool(RANKING.search(text))


def validate_named_shocks(question, answer, sources, evidence):
    requested = []
    for match in NAMED_SHOCK.finditer(question):
        symbol, direction, magnitude = match.group("symbol", "direction", "pct")
        if symbol != symbol.upper() and not (
            re.search(r"\bif\s+$", question[:match.start()], re.I)
            or re.search(rf"\b{re.escape(symbol.upper())}\b", evidence.get("PORTFOLIO-SCOPE", ""))
        ):
            continue
        symbol = symbol.upper()
        pct = float(magnitude)
        if direction.lower().startswith(("fall", "fel", "drop", "declin")):
            pct = -pct
        prefix = f"Hypothetical {symbol} price move {pct:+g}%:"
        exact = {source for source, value in evidence.items() if value.startswith(prefix)}
        requested.append((symbol, pct, exact))
    cited_shocks = {source for source in sources if source == "SHOCK" or source.startswith("SHOCK-")}
    requested_shocks = {source for _, _, exact in requested for source in exact}
    for symbol, pct, exact in requested:
        if exact:
            if not exact.intersection(sources):
                raise ValueError(f"Cite the exact {symbol} {pct:+g}% scenario")
            continue
        symbol_pattern = rf"(?<![\w.-]){re.escape(symbol)}(?![\w-]|\.\w)"
        absence_sources = {
            source for source, value in evidence.items()
            if source == "PORTFOLIO-SCOPE" and not re.search(symbol_pattern, value, re.I)
        }
        tool_errors = {
            source for source, value in evidence.items()
            if source.startswith("TOOL-ERROR-")
            and value.startswith(f"Scenario for {symbol} at {pct:+g}% is unavailable.")
        }
        gaps = absence_sources | tool_errors | ({"SCENARIO-COVERAGE"} & evidence.keys())
        cited_gaps = gaps.intersection(sources)
        absence_sources |= {
            source for source in tool_errors if "absent from the latest snapshot" in evidence[source]
        }
        numeric_evidence = " ".join([f"{abs(pct):g}", *(evidence[source] for source in cited_gaps | cited_shocks)])
        supported_numbers = {Decimal(number.replace(",", "")) for number in NUMBER.findall(numeric_evidence)}
        answer_text = re.sub(
            r"(?<![\w-])(?:" + "|".join(re.escape(source) for source in sources) + r")(?![\w-])",
            "",
            answer,
        )
        if (
            not cited_gaps
            or not re.search(symbol_pattern, answer_text, re.I)
            or not re.search(r"\b(?:no|missing|unavailable|cannot|not|absent)\b", answer_text, re.I)
            or (ABSENCE_CLAIM.search(answer_text) and not absence_sources.intersection(sources))
            or (not cited_shocks and SCENARIO_RESULT.search(answer_text))
            or cited_shocks - requested_shocks
            or any(
                Decimal(number.replace(",", "")) not in supported_numbers
                for number in NUMBER.findall(answer_text)
            )
        ):
            raise ValueError(f"No evidence for the requested {symbol} {pct:+g}% scenario")


def parse_answer(content, evidence, *, question=""):
    raw = ""
    try:
        parts = response_parts(content)
        # Preserve malformed visible parts even if the text cannot be joined.
        raw = json.dumps(parts)
        visible_text = "".join(part.get("text", "") for part in parts).strip()
        if visible_text:
            raw = visible_text
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.S)
        output = json.loads(fenced[1] if fenced else raw)
        answer, sources = output["answer"], output["sources"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Missing answer")
        if not isinstance(sources, list) or not sources or any(
            not isinstance(source, str) or source not in evidence for source in sources
        ):
            raise ValueError("Invalid citations")
        if ranking_needs_comparison(f"{question}\n{answer}") and "ATTRIBUTION-RANKING" not in sources:
            raise ValueError("Ranking claims require ATTRIBUTION-RANKING evidence")
        validate_named_shocks(question, answer, sources, evidence)
        return answer, list(dict.fromkeys(sources))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ExplanationUnavailable(
            f"Gemini answer validation failed: {error}.", raw_response=raw, evidence=evidence
        ) from None


def fixed_workflow(question, evidence, session):
    try:
        content = session.generate({
            "contents": [{"role": "user", "parts": [{
                "text": ANSWER_INSTRUCTIONS + "\n" + json.dumps({"question": question, "evidence": evidence})
            }]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": ANSWER_SCHEMA},
        })
    except ExplanationUnavailable as error:
        error.evidence = dict(evidence)
        raise
    answer, sources = parse_answer(content, evidence, question=question)
    return {"answer": answer, "sources": sources, "evidence": evidence, "tool_calls": []}


def explain(question, evidence, api_key, model=DEFAULT_MODEL):
    result = fixed_workflow(question, evidence, GeminiSession(api_key, model))
    return result["answer"], result["sources"]
