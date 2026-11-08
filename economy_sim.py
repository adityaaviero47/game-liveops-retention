"""Game economy simulation for LiveOps: what does a double-rewards event do to a soft-currency economy,
and which sink keeps it balanced?

This is a SIMULATION, not observed data. Player activity comes from the real WoW telemetry (each simulated
player's chance of playing on a given day is drawn from the real distribution of days active per 28 days);
the economy rules (coins earned per day, item prices, buying behaviour) are stated assumptions below.
Run after run_analysis.py:  python economy_sim.py"""
import json, numpy as np, pandas as pd

F = pd.read_csv("results/churn_features.csv")
play_p = (F.days_active_28d / 28).clip(0.02, 1).to_numpy()          # real per-player daily play probabilities

# ---- economy rules (ASSUMPTIONS) ----
A = dict(players=5000, days=60, coins_per_play_day=100,      # source: daily quests
         item_price=600, buy_prob_if_affordable=0.35,           # sink: regular shop
         event_days=(30, 37), event_multiplier=2.0,             # LiveOps event: double rewards for 7 days
         event_shop_price=1500, event_shop_buy_prob=0.5)        # optional extra sink during the event
SEED = 7

def simulate(event=False, event_shop=False):
    rng = np.random.default_rng(SEED)                            # same players and dice in every scenario
    p = rng.choice(play_p, A["players"])
    bal = np.zeros(A["players"]); src = snk = 0.0; daily = []
    for day in range(A["days"]):
        plays = rng.random(A["players"]) < p
        buy_roll, shop_roll = rng.random(A["players"]), rng.random(A["players"])
        in_event = event and A["event_days"][0] <= day < A["event_days"][1]
        earn = plays * A["coins_per_play_day"] * (A["event_multiplier"] if in_event else 1)
        bal += earn; src += earn.sum()
        buy = plays & (bal >= A["item_price"]) & (buy_roll < A["buy_prob_if_affordable"])
        bal -= buy * A["item_price"]; snk += buy.sum() * A["item_price"]
        if in_event and event_shop:
            eb = plays & (bal >= A["event_shop_price"]) & (shop_roll < A["event_shop_buy_prob"])
            bal -= eb * A["event_shop_price"]; snk += eb.sum() * A["event_shop_price"]
        daily.append(dict(day=day, median_balance=float(np.median(bal)), mean_balance=float(bal.mean())))
    d = pd.DataFrame(daily)
    return dict(coins_created=float(src), coins_removed=float(snk), sink_to_source=float(snk / src),
                mean_balance_end=float(bal.mean()), mean_balance_day_event_end=float(d.mean_balance[A["event_days"][1] - 1]),
                share_with_1500_plus=float((bal >= 1500).mean())), d

base, d0 = simulate()
ev, d1 = simulate(event=True)
evs, d2 = simulate(event=True, event_shop=True)
R = dict(assumptions=A, note="simulation; activity from real WoW players, economy rules assumed",
         baseline=base, double_rewards=ev, double_rewards_with_event_shop=evs)
R["impact"] = dict(
    event_extra_coins_share=ev["coins_created"] / base["coins_created"] - 1,
    balance_inflation_at_end=ev["mean_balance_end"] / base["mean_balance_end"] - 1,
    balance_inflation_with_shop=evs["mean_balance_end"] / base["mean_balance_end"] - 1,
    sink_ratio_base=base["sink_to_source"], sink_ratio_event=ev["sink_to_source"], sink_ratio_event_shop=evs["sink_to_source"],
    # at the last day of the event: how much higher are balances than without the event, and how much of that surplus the shop absorbs
    balance_spike_at_event_end=ev["mean_balance_day_event_end"] / base["mean_balance_day_event_end"] - 1,
    shop_absorbs_share_of_spike=1 - (evs["mean_balance_day_event_end"] - base["mean_balance_day_event_end"])
                                    / (ev["mean_balance_day_event_end"] - base["mean_balance_day_event_end"]))
pd.concat([d0.assign(scenario="baseline"), d1.assign(scenario="double_rewards"), d2.assign(scenario="double_rewards_event_shop")]).to_csv("results/economy_sim_daily.csv", index=False)
json.dump(R, open("results/economy_sim.json", "w"), indent=2)
print(json.dumps(R["impact"], indent=2))
