import os

import streamlit as st

from agent import agent_workflow
from gemini import APP_PACER, DEFAULT_MODEL, ExplanationUnavailable, GeminiSession, fixed_workflow
from risk import fixed_evidence_for, investigate, load_demo, shock_position

st.set_page_config(page_title="Financial Risk Intelligence Agent", layout="wide")
st.title("Financial Risk Intelligence Agent")
st.caption("Asia Equity · Synthetic portfolio · USD")

navs, snapshots, policy = load_demo()
report = investigate(navs, snapshots, policy)
st.subheader(f"Risk review · {report['current_date']}")
gross, limit, status = st.columns(3)
gross.metric(
    "Gross exposure",
    f"{report['current_gross']:g}% of NAV",
    f"{report['current_gross'] - report['previous_gross']:+g} pp",
    delta_color="off",
)
limit.metric("Gross exposure limit", f"{report['limit']:g}% of NAV")
status.metric("Policy status", "Limit breached" if report["breach"] else "Within limit")

st.subheader("Position contributions")
st.table([
    {"Position": row["symbol"], "Change (pp)": row["change_pp"], "Evidence": row["source"]}
    for row in report["contributions"]
])

date = report["current_date"]
positions = snapshots[date]
long = max((symbol for symbol in positions if positions[symbol] > 0), key=positions.get)
scenario = shock_position(positions, navs[date], long, -5)
st.subheader(f"What if {long} falls 5%?")
pnl, nav, exposure = st.columns(3)
pnl.metric("Hypothetical P&L", f"${scenario['pnl']:,.0f}")
nav.metric("Hypothetical NAV", f"${scenario['nav']:,.0f}")
exposure.metric("Hypothetical gross exposure", f"{scenario['gross_pct']:.2f}%")
st.caption("Other positions are held constant. Scenario P&L flows directly into NAV.")

evidence = fixed_evidence_for(report, positions, navs[date], long)
st.subheader("Ask Gemini")
mode = st.radio("Workflow", ["Fixed evidence", "ReAct tool calling"], horizontal=True)
question = st.text_input(
    "Analyst question",
    "Which position contributed most, and would the fund still breach if that position fell 5%?",
)
if st.button("Generate explanation", type="primary", disabled=not question.strip()):
    with st.spinner("Gemini is reviewing the evidence"):
        session = None
        try:
            session = GeminiSession(
                os.getenv("GEMINI_API_KEY"), os.getenv("GEMINI_MODEL", DEFAULT_MODEL), pacer=APP_PACER
            )
            if mode == "Fixed evidence":
                result = fixed_workflow(question, evidence, session)
            else:
                result = agent_workflow(question, report, positions, navs[date], session)
        except ExplanationUnavailable as error:
            st.warning(str(error))
            if session is not None:
                with st.expander("Explanation diagnostics"):
                    st.json({
                        "available_evidence": error.evidence,
                        "tool_calls": error.tool_calls,
                        "rejected_response": error.raw_response,
                        "api_errors": session.api_errors,
                    })
        else:
            st.text(result["answer"])
            for source in result["sources"]:
                statement = result["evidence"][source].replace("$", r"\$")
                st.caption(f"{source} · {statement}")
            metrics = session.metrics()
            tokens = metrics["total_tokens"] if metrics["total_tokens"] is not None else "unreported"
            st.caption(f"{metrics['api_calls']} API calls · {metrics['latency_ms']} ms · {tokens} tokens")
            if result["tool_calls"]:
                with st.expander("Tool calls"):
                    st.json(result["tool_calls"])

with st.expander("View all evidence"):
    for source, statement in evidence.items():
        statement = statement.replace("$", r"\$")
        st.markdown(f"**{source}** · {statement}")
