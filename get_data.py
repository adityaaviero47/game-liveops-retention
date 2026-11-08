"""Download the 2008 WoW Avatar History sample and write the compressed copy the analysis reads.
Only needed if data/ is empty:  python get_data.py"""
import os, pandas as pd

# characters with IDs 1-10,000 from the WoWAH dataset (Lee et al., 2011; Kaggle "warcraft-avatar-history"),
# as published in a public GitHub repository
URL = ("https://raw.githubusercontent.com/davide97l/WoW-dataset-analysis/be34ea97c122411772ca5aace226869c83e4b7d3/"
       "warcraft-avatar-history/wowah_data_filtered_10000.csv")

def build(src=URL):
    d = pd.read_csv(src, skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["zone"] = d.zone.str.strip()
    os.makedirs("data", exist_ok=True)
    d.to_csv("data/wow_sessions_2008.csv.gz", index=False, compression="gzip")
    print(f"data/wow_sessions_2008.csv.gz  {len(d):,} rows")

if __name__ == "__main__":
    build()
