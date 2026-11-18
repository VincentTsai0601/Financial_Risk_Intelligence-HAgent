import csv
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"


def load_demo():
    with (DATA / "portfolio.csv").open(newline="", encoding="utf-8") as file:
        navs = {row["snapshot"]: int(row["nav"]) for row in csv.DictReader(file)}
    snapshots = {date: {} for date in navs}
    with (DATA / "positions.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            snapshots[row["snapshot"]][row["symbol"]] = int(row["market_value"])
    policy = json.loads((DATA / "policy.json").read_text(encoding="utf-8"))
    return navs, snapshots, policy


def gross_exposure(positions, nav):
    if nav <= 0:
        raise ValueError("NAV must be positive to calculate gross exposure.")
    return 100 * sum(abs(value) for value in positions.values()) / nav


def investigate(navs, snapshots, policy):
    previous, current = sorted(snapshots)
    before, after = snapshots[previous], snapshots[current]
    previous_gross = gross_exposure(before, navs[previous])
    current_gross = gross_exposure(after, navs[current])
    limit = policy["gross_limit_pct"]
    contributions = []
    evidence = {
        policy["id"]: policy["text"],
        "EXPOSURE": (
            f"Asia Equity gross exposure: {previous_gross:g}% on {previous} and "
            f"{current_gross:g}% on {current}. Change: {current_gross - previous_gross:+g} pp. "
            f"Current status: {'breach' if current_gross > limit else 'within limit'}."
        ),
    }
    for symbol in sorted(before.keys() | after.keys()):
        old, new = before.get(symbol, 0), after.get(symbol, 0)
        change = 100 * (abs(new) / navs[current] - abs(old) / navs[previous])
        source_id = f"POSITION-{symbol}"
        contributions.append({"symbol": symbol, "change_pp": round(change, 8), "source": source_id})
        evidence[source_id] = (
            f"positions.csv: {symbol}, {previous} value ${old:,}, "
            f"{current} value ${new:,}. portfolio.csv: NAV ${navs[previous]:,} "
            f"and ${navs[current]:,}, respectively. Exposure contribution: {change:+g} pp."
        )
    contributions.sort(key=lambda item: -abs(item["change_pp"]))
    ranked = sorted(contributions, key=lambda item: -item["change_pp"])
    evidence["ATTRIBUTION-RANKING"] = (
        f"Complete position comparison for {previous} to {current}, sorted by signed "
        "contribution to the increase in gross exposure: "
        + "; ".join(f"{row['symbol']} {row['change_pp']:+g} pp" for row in ranked)
        + ". Based on positions.csv and portfolio.csv. Ties have equal rank."
    )
    evidence["PORTFOLIO-SCOPE"] = (
        f"Complete current position list on {current}: " + ", ".join(sorted(after))
        + ". Other symbols have no position data in this demo."
    )
    evidence["NOTES"] = "The demo snapshots contain no trade rationale or analyst commentary."
    return {
        "current_date": current,
        "current_gross": current_gross,
        "previous_gross": previous_gross,
        "limit": limit,
        "breach": current_gross > limit,
        "contributions": contributions,
        "evidence": evidence,
    }


def shock_position(positions, nav, symbol, shock_pct):
    original = positions[symbol]
    pnl = original * shock_pct / 100
    shocked = {**positions, symbol: original + pnl}
    shocked_nav = nav + pnl
    return {
        "pnl": pnl,
        "nav": shocked_nav,
        "gross_pct": gross_exposure(shocked, shocked_nav),
    }


def scenario_evidence(positions, nav, symbol, shock_pct, limit):
    result = shock_position(positions, nav, symbol, shock_pct)
    status = "would breach" if result["gross_pct"] > limit else "would remain within"
    return (
        f"Hypothetical {symbol} price move {shock_pct:+g}%: P&L {result['pnl']:+,.2f} USD, "
        f"NAV {result['nav']:,.2f} USD, gross exposure {result['gross_pct']:.2f}% of NAV. "
        f"The portfolio {status} the {limit:g}% limit. "
        "Other positions are held constant and P&L flows into NAV."
    )


def fixed_evidence_for(report, positions, nav, symbol):
    return {
        **report["evidence"],
        "SCENARIO-COVERAGE": (
            f"Fixed evidence includes only the precomputed {symbol} -5% scenario. "
            "The scenario tool is unavailable in the fixed-evidence workflow."
        ),
        "SHOCK": scenario_evidence(positions, nav, symbol, -5, report["limit"]),
    }
