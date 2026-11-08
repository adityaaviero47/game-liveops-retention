"""A working LiveOps retention agent, backtested on months the churn model never saw (Aug-Oct 2008).
Each run: score every active player -> pick an action from the player's own churn signals -> apply guardrails
(contact cap, budget cap, a fixed 10% holdout, one action per player per run) -> log every decision -> evaluate.
The historical data has no interventions, so the evaluation measures targeting quality and guardrail compliance;
the effect of each action has to be measured live against the holdout.
Run after run_analysis.py:  python liveops_agent.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

F = pd.read_csv("results/churn_features.csv", parse_dates=["cutoff"])
FEATS = ["hours_28d", "days_active_28d", "hours_7d", "sessions_28d", "level", "in_guild", "zones_28d", "at_cap", "trend", "days_since_last"]
tr = F[F.cutoff <= "2008-07-01"]
model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=7).fit(tr[FEATS], tr.churn)

GUARDRAILS = dict(max_contact_share=0.20,    # never contact more than 20% of active players in one run
                  max_reward_actions=60,     # budget: at most 60 reward-carrying actions per run
                  holdout_share=0.10,        # fixed random holdout, never contacted
                  min_risk=0.30)             # do not act below this churn probability
REWARD_ACTIONS = {"comeback_reward", "catch_up_boost"}

def choose_action(p):
    """Rule-based policy: the action follows the strongest churn signal for this player."""
    if p.days_since_last > 14: return "comeback_reward", "away 14+ days"
    if p.in_guild == 0: return "guild_invite", "not in a guild"
    if p.trend < 0.5: return "catch_up_boost", "play falling"
    return "check_in_message", "high risk, no single signal"

def run(cutoff, frame):
    rng = np.random.default_rng(int(cutoff.strftime("%Y%m%d")))
    X = frame.copy()
    X["risk"] = model.predict_proba(X[FEATS])[:, 1]
    X["holdout"] = rng.random(len(X)) < GUARDRAILS["holdout_share"]
    cap = int(len(X) * GUARDRAILS["max_contact_share"])
    log, rewards = [], 0
    for _, p in X.sort_values("risk", ascending=False).iterrows():
        if p.risk < GUARDRAILS["min_risk"] or len([l for l in log if l["decision"] == "act"]) >= cap: break
        action, reason = choose_action(p)
        if p.holdout: decision = "holdout"
        elif action in REWARD_ACTIONS and rewards >= GUARDRAILS["max_reward_actions"]: action, decision = "check_in_message", "act"; reason += " (reward budget used up)"
        else: decision = "act"
        if decision == "act" and action in REWARD_ACTIONS: rewards += 1
        log.append(dict(cutoff=str(cutoff.date()), char=int(p.char), risk=round(float(p.risk), 4), action=action, reason=reason,
                        decision=decision, churned=int(p.churn)))
    return pd.DataFrame(log), X

logs, evals = [], []
for cut in pd.to_datetime(["2008-08-01", "2008-09-01", "2008-10-01"]):
    fr = F[F.cutoff == cut]
    L, X = run(cut, fr)
    acted, held = L[L.decision == "act"], L[L.decision == "holdout"]
    # guardrail checks: the run fails loudly if any is broken
    assert len(acted) <= int(len(X) * GUARDRAILS["max_contact_share"])
    assert acted.action.isin(REWARD_ACTIONS).sum() <= GUARDRAILS["max_reward_actions"]
    assert not acted.char.isin(X.char[X.holdout]).any() and not acted.char.duplicated().any()
    evals.append(dict(cutoff=str(cut.date()), active_players=len(X), contacted=len(acted), held_out_flagged=len(held),
                      contact_share=len(acted) / len(X),
                      precision=float(acted.churned.mean()),                      # share of contacted who would have churned
                      recall=float(L.churned.sum() / X.churn.sum()),              # share of all churners the agent flagged
                      base_churn=float(X.churn.mean()), reward_actions=int(acted.action.isin(REWARD_ACTIONS).sum()),
                      holdout_churn=float(held.churned.mean()) if len(held) else None,
                      action_mix=acted.action.value_counts().to_dict(), guardrail_violations=0))
    logs.append(L)
logs = pd.concat(logs)
E = pd.DataFrame(evals)
summary = dict(guardrails=GUARDRAILS, runs=evals,
               overall=dict(decisions_logged=int(len(logs)), contacted=int((logs.decision == "act").sum()),
                            precision=float(logs[logs.decision == "act"].churned.mean()),
                            recall=float(logs.churned.sum() / F[F.cutoff.isin(pd.to_datetime(["2008-08-01", "2008-09-01", "2008-10-01"]))].churn.sum()),
                            base_churn=float(E.base_churn.mean()), lift=float(logs[logs.decision == "act"].churned.mean() / E.base_churn.mean()),
                            guardrail_violations=0,
                            action_mix=logs[logs.decision == "act"].action.value_counts().to_dict()))
logs.to_csv("results/agent_decision_log.csv", index=False)
json.dump(summary, open("results/agent_eval.json", "w"), indent=2, default=str)
print(json.dumps(summary["overall"], indent=2, default=str))
