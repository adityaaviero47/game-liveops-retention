"""Charts for README from results/results.json and results/daily_active_players.csv."""
import json, matplotlib, pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))
dau = pd.read_csv("results/daily_active_players.csv", index_col=0, parse_dates=True).dau
fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
ax[0, 0].plot(dau.index, dau.rolling(7, min_periods=4).mean(), color="#0052CC")
for day, lab, yy, ha in [(R["step_up"]["day"], "Step-up\n(+46% DAU) ", 75, "right"), (R["launch"]["launch_day"], " Expansion\n launch", 75, "left")]:
    ax[0, 0].axvline(pd.Timestamp(day), color="#E07A2F", ls="--", lw=1); ax[0, 0].text(pd.Timestamp(day), yy, lab, fontsize=8, ha=ha)
ax[0, 0].set_ylabel("Daily active players (7-day average)"); ax[0, 0].set_ylim(0, 400)
ax[0, 0].set_title("2008 activity: the October step-up beat the launch")

r = R["retention"]["active_in"]; ms = list(r)
ax[0, 1].plot(range(len(ms)), [r[m] * 100 for m in ms], "o-", color="#0052CC")
ax[0, 1].set_xticks(range(len(ms))); ax[0, 1].set_xticklabels([m[5:] for m in ms])
ax[0, 1].set_ylim(0, 105); ax[0, 1].set_xlabel("Month of 2008"); ax[0, 1].set_ylabel("January players still active (%)")
ax[0, 1].set_title(f"Retention: {r[ms[-1]]*100:.0f}% of January players remain in December")

c = R["churn"]
labs = ["In a guild", "No guild", "Steady play", "Play falling\n(last week < half)"]
vals = [c["churn_in_guild"], c["churn_no_guild"], c["churn_steady"], c["churn_falling"]]
ax[1, 0].bar(labs, [v * 100 for v in vals], color=["#0052CC", "#E07A2F", "#0052CC", "#E07A2F"])
for i, v in enumerate(vals): ax[1, 0].text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center")
ax[1, 0].set_ylabel("Churn in the next 30 days (%)"); ax[1, 0].set_ylim(0, 50)
ax[1, 0].set_title("Churn signals: no guild 2.6x, falling play 3.8x")

L = R["launch"]
ax[1, 1].bar(["Ordinary month\n(baseline)", "Expansion\nlaunch", "October\nstep-up"],
             [L["baseline_reactivation_14d"] * 100, L["reactivated_14d"] * 100, R["step_up"]["reactivated_14d"] * 100], color=["#9AA5AD", "#0052CC", "#0052CC"])
for i, v in enumerate([L["baseline_reactivation_14d"], L["reactivated_14d"], R["step_up"]["reactivated_14d"]]): ax[1, 1].text(i, v * 100 + .3, f"{v*100:.1f}%", ha="center")
ax[1, 1].set_ylabel("Lapsed players back within 14 days (%)"); ax[1, 1].set_ylim(0, 16)
ax[1, 1].set_title("Reactivation of players away 30+ days")
for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("results/summary.png", dpi=150)
