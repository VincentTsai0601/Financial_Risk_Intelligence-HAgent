# Financial Risk Intelligence Agent

A portfolio project for a risk analyst's daily review. The Streamlit app detects a gross-exposure breach, attributes the change to positions, and uses Gemini to explain the result with cited evidence. All data and policy examples are synthetic.

## Run on Ubuntu

From the extracted project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
read -rsp "Gemini API key: " GEMINI_API_KEY
echo
export GEMINI_API_KEY
python -m streamlit run app.py
```

Open http://localhost:8501. If creating the environment reports missing ensurepip, install `python3-venv` with apt. Set `GEMINI_MODEL` to select another model available to your API key. The default is `gemini-3.5-flash-lite`. The dashboard also runs without an API key.

For Windows PowerShell, create the environment with `py -3 -m venv .venv`, use `.\.venv\Scripts\python.exe` for Python commands, and set the key with `$env:GEMINI_API_KEY`.

## Run with Docker Compose

On Ubuntu with Docker Engine and Compose v2, build and start the CPU-only dashboard:

```bash
docker compose up --build --wait app
```

Open http://127.0.0.1:8501. Set `APP_PORT` before starting Compose if port 8501 is in use. To enable Gemini explanations, export `GEMINI_API_KEY` before starting Compose or set it in a local, ignored `.env` file. The build context excludes credentials and evaluation CSVs.

To run the optional GPU judge, use a host with a working NVIDIA driver and NVIDIA Container Toolkit. The first startup downloads Qwen3-4B and caches its weights in a named volume.

```bash
docker compose --profile judge up -d judge
docker compose logs -f judge
# Once the server is ready, run these from the host:
curl --fail http://127.0.0.1:8001/v1/models
python judge.py evaluation.csv --output evaluation-docker-judged.csv
```

Use `VLLM_IMAGE` to select a different vLLM image tag for your host. Stop services with `docker compose --profile judge down`. The model cache persists across service restarts.

On pushes and pull requests, GitHub Actions runs the unit tests, builds and starts the app container, checks its HTTP health endpoint, and verifies that the bundled demo loads with both workflow options. CI uses no Gemini credentials or GPU.

## Two-minute demo

1. Open the dashboard. Gross exposure rises from 145% to 165% of NAV, exceeding the synthetic 160% limit.
2. Open position evidence. ALPHA accounts for +15 percentage points and BETA for +5.
3. Ask Gemini why the portfolio was flagged, then open the cited evidence. Ask why the trades were made to demonstrate how missing rationale is handled.
4. Review the 5% ALPHA price shock. P&L is −$45,000 and NAV falls to $955,000. Gross exposure rises to 168.06% because the NAV denominator shrinks. Ask Gemini to explain the scenario.

Gross exposure is the sum of absolute position market values divided by NAV. The attribution subtracts each position's prior NAV weight from its current NAV weight, using absolute values for both. The shock holds other positions fixed and includes the shocked position's P&L in NAV.

## Tests

```text
python -m unittest discover -s tests -v
```

Tests cover the breach, threshold equality, attribution, scenario, mocked Gemini response, unsupported source IDs, timeout, and missing credentials. Citation validation checks that the referenced IDs exist. Review the displayed evidence to verify that it supports the explanation. To verify the live integration, run the demo with your Gemini key.

## Compare the two Gemini workflows

Choose **Fixed evidence** for the one-call workflow or **ReAct tool calling** to let Gemini request verified exposure, attribution, and hypothetical scenarios. Both use the same Python calculations. The agent keeps the model's tool-call history, validates tool inputs, and shows the calls and evidence behind the answer. It starts with a required exposure lookup, then lets Gemini select further tools. This prevents the initial notes-only answer from treating a policy breach as missing trade rationale. It allows up to six tool calls across four tool rounds.

Try “Which position contributed most, and would the fund still breach if that position fell 5%?” The expected answer is ALPHA (+15 pp), with gross exposure still above the 160% limit at 168.06%.

Run the 12-case paired evaluation after setting your API key:

```bash
python evaluate.py --output evaluation.csv
```

This sends at least 24 requests if all cases complete, plus the tool loop's follow-up requests. Use `--repeats 3` to check repeatability. Choose a fresh output path each time. The runner waits two seconds between workflow runs (adjust with `--delay`) and spaces all HTTP attempts by at least six seconds (adjust with `--request-interval`). The shared pacer covers tool turns and retries as well as the first request. Gemini sessions in the same process share the six-second interval by default, including standalone explanations. Separate processes have separate pacing. Choose an interval using your project's actual per-minute limits. For example, 15 seconds gives at most about four request starts per minute from this sequential runner. It cannot resolve a daily quota or a spending limit.

Transient Gemini HTTP 429/5xx failures receive up to two retries. A 429 uses 30 then 60 seconds of backoff, honoring a longer numeric provider retry delay up to 60 seconds. The client stops retries when the provider identifies a daily limit or asks for a longer wait. The CSV records provider and connection errors per attempt in `api_errors`, including retries that recover. The client redacts the API key from provider diagnostics. Latency now includes pacing and backoff, recorded separately as `wait_ms`; account for that when comparing with earlier timing results. The CSV records answers, cited passages, tool calls, latency, provider-reported token use, and errors. Score the two review columns as 1 or 0 against the expected answer and cited evidence, or run the local judge below. Count a citation as supporting an answer only when it substantiates each material claim. Treat failed calls as failures in the overall success denominator and report their count.

The fixed baseline receives the original snapshots, policy, and precomputed 5% ALPHA scenario. Report shared-evidence cases separately from the BETA and GAMMA shock cases, which test the agent's ability to compute a scenario on demand. For a supported question with insufficient data, a clear statement of the gap earns credit. The CSV also records workflow version, all available evidence, rejected answer text, and provider error details. Rejected text excludes thinking parts and encrypted thought signatures. Incomplete Gemini candidates retain their finish reason and any visible partial reply. Tool traces collected before a failure are retained with tool inputs, results or errors, and provider call IDs when available. The dashboard exposes failed explanations under **Explanation diagnostics**. The source-existence checks and ranking-citation check are deterministic. Numeric and general citation quality are still reviewed by a human or the local judge.

A ranking claim must cite the complete `ATTRIBUTION-RANKING` comparison, which Python derives from every position's signed contribution. The ranking check recognizes common English phrases in both the question and answer. For explicit requests such as “BETA rises 10%,” validation checks that the cited scenario matches the position and move, or that the answer cites the specific gap. Human review still checks semantic support for other phrasing and whether cited evidence supports the conclusions. `PORTFOLIO-SCOPE` and `SCENARIO-COVERAGE` give the fixed workflow sources for explaining missing positions or uncomputed scenarios. A justified abstention on a question expecting a computed scenario still earns zero answer accuracy, while its citation can be assessed independently if it passes validation.

In the prior Gemini 3.5 Flash-Lite evaluation, ReAct completed 10 of 12 attempts and Fixed evidence completed 7 of 12. Each had two HTTP 429 failures. Fixed evidence also had three citation validation failures. ReAct computed the BETA +10% shock (−$55,000 P&L, 180.42% gross exposure) and the GAMMA +10% shock (+$20,000 P&L, 163.73% gross exposure). Both answers to the paraphrased ranking question received full Qwen citation credit despite omitting comparative evidence. These are historical results from before the current checks.

References: [ReAct variants on Gemini 2.5 Flash](https://aclanthology.org/2026.acl-srw.82/), [FinToolBench](https://arxiv.org/abs/2603.08262), and [ReliabilityBench](https://arxiv.org/abs/2601.06112).

## Judge answers with local Qwen3-4B

This is a FinToolBench-inspired evaluation with Qwen3-4B as the judge. FinToolBench used GPT-5.1 for judging. Treat the local judge's scores as estimates and check a sample against human review.

Run the server in a separate Ubuntu terminal. If you already have a working vLLM environment, activate it. Otherwise create a dedicated environment for its CUDA dependencies:

```bash
python3 -m venv ~/.venvs/risk-judge
source ~/.venvs/risk-judge/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm

vllm serve Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype half \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90
```

The first launch downloads the model. The 8,192-token context and one concurrent sequence are starting settings for a 24 GB GPU with available VRAM. Check `nvidia-smi` for other running workloads. If the KV cache cannot fit after those workloads stop, try `--max-model-len 4096`. Installing vLLM requires a supported Python/CUDA/driver combination. See the official [Qwen vLLM deployment guide](https://qwen.readthedocs.io/en/stable/deployment/vllm.html).

Keep the server running. In the project terminal, using the application's environment:

```bash
source .venv/bin/activate
curl --fail http://127.0.0.1:8001/v1/models
python judge.py evaluation.csv --output evaluation-judged.csv
```

The judge requires no Gemini key. It uses vLLM's local Chat Completions endpoint, schema-constrained JSON, and Qwen's `enable_thinking=false` setting. No additional client package is required. The default judge model name is `qwen3-4b`. The model name must match the server's served name. Override the address with `--base-url http://127.0.0.1:8001/v1` and the name with `--model qwen3-4b`. An authenticated server can use `JUDGE_API_KEY`.

The output contains the original evaluation fields plus:
- `judge_answer_correct` and `judge_citations_support_answer`, scored 0 or 1.
- `judge_verdicts`, containing each verdict and its reasons.
- Judge model, prompt version, latency, token use, and status.

The manual scoring columns are retained. Questions, reference answers and cited passages are passed to the judge with the generating model and workflow omitted. Citation scoring asks whether the supplied passages support all material factual claims. Reference numbers come from the demo's Python calculations and expected-answer fixtures.

Rows with generation errors receive zero scores without a judge call. Judge outages or invalid responses leave scores blank with `judge_status=judge_error`, and the command exits with status 1. The summary reports generation and judge failures separately and shows scores both across all attempts and among scored answers. When judge failures leave a group incomplete, it reports the scored subset and withholds the all-attempt scores. The `judge_rejected_response` column records the rejected verdict text or HTTP error body, and `judge_error` identifies the validation or API failure. For ranking cases in the bundled demo, the program verifies the complete ranking passage or every position's contribution passage even when Qwen awards full credit. The original Qwen verdict and the reason for any override are both retained.

For repeatability, run `python judge.py evaluation.csv --output evaluation-judged-3.csv --repeats 3`. Scores then represent the mean of three binary judgments. Failed repeats retain partial verdicts but leave aggregate scores blank. Use a fresh output path for each run. This small synthetic evaluation supports a portfolio demo. It has its own questions and tools, and does not establish performance on the FinToolBench dataset.

The local tests cover the request format, invalid scores, output truncation, repeated scoring, generation failures, judge outages, and preservation of manual review fields. Actual Qwen judgment quality and GPU operation require a live run on your machine.

## Rerun after the reliability changes

Keep your previous CSV files for comparison. With the Qwen3-4B server running in the first terminal, run these commands in the project terminal.

```bash
source .venv/bin/activate
export GEMINI_MODEL=gemini-3.5-flash-lite
python evaluate.py --output evaluation-v4.csv --request-interval 15
python judge.py evaluation-v4.csv --model qwen3-4b \
  --output evaluation-v4-judged.csv
```

Set `GEMINI_API_KEY` in that terminal if needed. The local judge rubric is version `risk-judge-v4`. It assesses whether evidence supports a stated coverage gap independently from whether the answer provides the requested result. The bundled judged CSV used v3 and recorded 21 invalid verdicts out of 24 rows. Rerun the judge on an old evaluation CSV with a fresh output filename to compare rubric changes without generating new Gemini answers. Retain the model name and rubric version when reporting results.

The synthetic tests verify the new ranking source, preservation of rejected answers, pacing across sessions, provider retry delays and daily-quota handling. The actual decrease in API failures and improvement in citation support require a new live evaluation on your machine.
