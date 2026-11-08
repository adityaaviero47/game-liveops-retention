"""Game LiveOps retention analysis on real player telemetry: World of Warcraft Avatar History (WoWAH), 2008.
Every 10 minutes the dataset recorded each online character's level, zone and guild on one server.
This script measures player retention, the impact of a major content launch (the Wrath of the Lich King
expansion, which raised the level cap from 70 to 80) on activity and on lapsed players, which signals predict churn, and an early-warning model
a LiveOps agent could act on. Reproduces every number in README.md:  python run_analysis.py"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

R = {}
d = pd.read_csv("data/wow_sessions_2008.csv.gz")
d["ts"] = pd.to_datetime(d.timestamp, format="%m/%d/%y %H:%M:%S")

# ---------- 1. data checks ----------
R["validation"] = dict(snapshots=len(d), exact_duplicates=int(d.duplicated().sum()), missing=int(d.isna().sum().sum()),
                       level_out_of_range=int((~d.level.between(1, 80)).sum()))
d = d.drop_duplicates().copy()
d["day"] = d.ts.dt.normalize()
d["in_guild"] = d.guild >= 0
R["data"] = dict(snapshots=len(d), players=int(d.char.nunique()), start=str(d.ts.min().date()), end=str(d.ts.max().date()),
                 days_recorded=int(d.day.nunique()), hours_played=float(len(d) / 6))
# days with no snapshots at all are recording gaps on the dataset's side, not player behaviour
all_days = pd.date_range("2008-01-01", "2008-12-31")
R["data"]["missing_days"] = int(len(all_days) - d.day.nunique())

# ---------- 2. sessions: consecutive 10-minute snapshots (gap <= 30 min) ----------
d = d.sort_values(["char", "ts"])
gap = d.groupby("char").ts.diff().dt.total_seconds().div(60)
d["session"] = (gap.isna() | (gap > 30)).groupby(d.char).cumsum()
s = d.groupby(["char", "session"]).agg(start=("ts", "min"), end=("ts", "max"), snaps=("ts", "size"))
s["hours"] = s.snaps / 6
R["sessions"] = dict(sessions=int(len(s)), median_session_hours=float(s.hours.median()), mean_session_hours=float(s.hours.mean()))

# ---------- 3. retention: share of January players still playing later ----------
d["month"] = d.ts.dt.to_period("M")
first = d.groupby("char").month.min()
active = d.groupby(["char", "month"]).size().unstack(fill_value=0) > 0
jan = active.index[first == pd.Period("2008-01", "M")]
R["retention"] = dict(january_players=int(len(jan)),
                      active_in={str(m): float(active.loc[jan, m].mean()) for m in active.columns})
mau = d.groupby("month").char.nunique()
R["mau"] = {str(k): int(v) for k, v in mau.items()}

# ---------- 4. content launch: Wrath of the Lich King (level cap 70 -> 80) ----------
# launch day on this realm = first day any character is above level 70 (taken from the data, not assumed)
LAUNCH = d.day[d.level > 70].min()
dau = d.groupby("day").char.nunique().reindex(all_days)
hrs = d.groupby("day").size().div(6).reindex(all_days)
# 21 days after launch vs the 21 days before it, skipping the day before launch (very low activity: launch-day maintenance)
pre = (all_days >= LAUNCH - pd.Timedelta(days=22)) & (all_days < LAUNCH - pd.Timedelta(days=1))
post = (all_days >= LAUNCH) & (all_days < LAUNCH + pd.Timedelta(days=21))
R["launch"] = dict(launch_day=str(LAUNCH.date()), dau_day_before=float(dau[LAUNCH - pd.Timedelta(days=1)]), dau_pre=float(dau[pre].mean()), dau_post=float(dau[post].mean()), dau_lift=float(dau[post].mean() / dau[pre].mean() - 1),
                   hours_per_dau_pre=float((hrs[pre] / dau[pre]).mean()), hours_per_dau_post=float((hrs[post] / dau[post]).mean()))
R["launch"]["hours_per_dau_lift"] = R["launch"]["hours_per_dau_post"] / R["launch"]["hours_per_dau_pre"] - 1
# reactivation: players who had played before but were away for the 30 days before a date; share back within 14 days
last_seen = lambda cut: d[d.ts < cut].groupby("char").ts.max()
def reactivation(event):
    ls = last_seen(event)
    lapsed = ls[ls < event - pd.Timedelta(days=30)].index
    back = d[(d.ts >= event) & (d.ts < event + pd.Timedelta(days=14))].char.unique()
    return len(lapsed), float(np.isin(lapsed, back).mean())
n_l, r_l = reactivation(LAUNCH)
# baseline: same rule on ordinary dates in the year (no content launch)
base = [reactivation(pd.Timestamp(x)) for x in ["2008-05-13", "2008-06-13", "2008-07-13", "2008-08-13", "2008-09-13"]]
r_b = float(np.mean([b[1] for b in base]))
# a second, unexplained step-up in activity: the largest jump in 7-day average DAU during the year
roll = dau.rolling(7, min_periods=5).mean()
jump = (roll.shift(-7) - roll).idxmax() + pd.Timedelta(days=1)                # first day of the 7 days that most exceed the 7 before
step_pre = dau[(all_days >= jump - pd.Timedelta(days=14)) & (all_days < jump)].mean()
step_post = dau[(all_days >= jump) & (all_days < jump + pd.Timedelta(days=14))].mean()
n_s, r_s = reactivation(jump)
R["step_up"] = dict(day=str(jump.date()), dau_14d_before=float(step_pre), dau_14d_after=float(step_post), dau_lift=float(step_post / step_pre - 1),
                    lapsed_players=n_s, reactivated_14d=r_s)
R["launch"].update(lapsed_players=n_l, reactivated_14d=r_l, baseline_reactivation_14d=r_b, reactivation_vs_baseline=r_l / r_b)
# levelling to the new cap: players at level 70 on launch day who reached 80 by 31 Dec
lvl_at = d[d.ts < LAUNCH + pd.Timedelta(days=1)].groupby("char")["level"].max()
at70 = lvl_at[lvl_at == 70].index
post_lvl = d[d.ts >= LAUNCH].groupby("char")["level"].max()
played_after = at70[np.isin(at70, post_lvl.index)]
reached = post_lvl.reindex(played_after) == 80
t80 = d[(d.ts >= LAUNCH) & (d.level == 80)].groupby("char").ts.min()
days80 = (t80.reindex(played_after[reached.to_numpy()]) - LAUNCH).dt.days
R["launch"].update(level70_at_launch=int(len(at70)), level70_played_after=int(len(played_after)),
                   reached_80_share=float(reached.mean()), median_days_to_80=float(days80.median()))

# ---------- 5. churn: who stops playing, and the warning signs ----------
# snapshot at the 1st of each month; churned = played in the 28 days before, then no play in the next 30 days
def frame(cut):
    h = d[(d.ts < cut) & (d.ts >= cut - pd.Timedelta(days=28))]
    if h.empty: return None
    g = h.groupby("char")
    X = pd.DataFrame(dict(
        hours_28d=g.size() / 6, days_active_28d=g.day.nunique(),
        hours_7d=h[h.ts >= cut - pd.Timedelta(days=7)].groupby("char").size() / 6,
        sessions_28d=h.groupby("char").session.nunique(), level=g["level"].max(), in_guild=g.in_guild.max().astype(int),
        zones_28d=g.zone.nunique())).fillna(0)
    X["at_cap"] = (X.level >= (80 if cut > LAUNCH else 70)).astype(int)
    X["trend"] = X.hours_7d / (X.hours_28d / 4).clip(lower=1 / 6)      # last week vs the 4-week weekly average
    X["days_since_last"] = (cut - g.ts.max()).dt.total_seconds() / 86400
    fut = d[(d.ts >= cut) & (d.ts < cut + pd.Timedelta(days=30))].char.unique()
    X["churn"] = (~X.index.isin(fut)).astype(int); X["cutoff"] = cut
    return X
cuts = pd.date_range("2008-02-01", "2008-11-01", freq="MS")
F = pd.concat([frame(c) for c in cuts]).reset_index().rename(columns={"index": "char"})
R["churn"] = dict(player_months=int(len(F)), churn_rate=float(F.churn.mean()),
                  churn_in_guild=float(F.churn[F.in_guild == 1].mean()), churn_no_guild=float(F.churn[F.in_guild == 0].mean()))
R["churn"]["no_guild_vs_guild"] = R["churn"]["churn_no_guild"] / R["churn"]["churn_in_guild"]
low = F.days_active_28d <= 3
R["churn"].update(churn_low_activity=float(F.churn[low].mean()), churn_high_activity=float(F.churn[F.days_active_28d >= 15].mean()),
                  low_activity_share=float(low.mean()))
fall = F.trend < 0.5
R["churn"].update(churn_falling=float(F.churn[fall].mean()), churn_steady=float(F.churn[~fall].mean()), falling_share=float(fall.mean()))
# early-warning model: train on Feb-Jul snapshots, test on Aug-Oct (later, unseen months)
feats = ["hours_28d", "days_active_28d", "hours_7d", "sessions_28d", "level", "in_guild", "zones_28d", "at_cap", "trend", "days_since_last"]
tr = F[F.cutoff <= "2008-07-01"]; te = F[(F.cutoff >= "2008-08-01") & (F.cutoff <= "2008-10-01")]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, random_state=7).fit(tr[feats], tr.churn)
p = m.predict_proba(te[feats])[:, 1]; y = te.churn.to_numpy()
top = np.argsort(-p)[: int(len(p) * .2)]
rule = -te.days_since_last.to_numpy()                                   # baseline rule: most recently active last
R["early_warning"] = dict(train_months="Feb-Jul 2008", test_months="Aug-Oct 2008", test_player_months=int(len(te)),
                          base_rate=float(y.mean()), auc=float(roc_auc_score(y, p)), top20_capture=float(y[top].sum() / y.sum()),
                          top20_churn_rate=float(y[top].mean()), days_since_last_rule_auc=float(roc_auc_score(y, -rule)))

dau.rename("dau").to_csv("results/daily_active_players.csv")
F.to_csv("results/churn_features.csv", index=False)
json.dump(R, open("results/results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
