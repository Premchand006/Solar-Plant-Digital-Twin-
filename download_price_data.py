import os
import pandas as pd
from nemosis import dynamic_data_compiler

print("Downloading SA1 dispatch prices 2022-2026 (using cache)...")
os.makedirs("price_cache", exist_ok=True)

# Pull WITHOUT filter first so we get all regions, then filter in pandas
data = dynamic_data_compiler(
    start_time       = "2022/01/01 00:00:00",
    end_time         = "2026/01/20 00:00:00",
    table_name       = "DISPATCHPRICE",
    raw_data_location= "price_cache",          # reuses already-downloaded cache
)

print(f"Total rows all regions: {len(data):,}")
print(f"Columns: {list(data.columns)}")
print(f"Regions: {sorted(data['REGIONID'].unique())}")

# Filter SA1 in pandas (not inside nemosis)
sa = data[data["REGIONID"] == "SA1"].copy()
print(f"SA1 rows: {len(sa):,}")

# Rename to match preprocess.py expectations
sa = sa.rename(columns={
    "SETTLEMENTDATE"     : "Settlement Date",
    "RRP"                : "Spot Price ($/MWh)",
})

# Keep only what we need
sa = sa[["Settlement Date", "Spot Price ($/MWh)"]].copy()
sa = sa.sort_values("Settlement Date").reset_index(drop=True)
sa["Spot Price ($/MWh)"] = pd.to_numeric(sa["Spot Price ($/MWh)"], errors="coerce")
sa = sa.dropna()

out = "NEMPRICEANDDEMAND_SA1_2022_2026.csv"
sa.to_csv(out, index=False)

print(f"\nSaved {len(sa):,} rows -> {out}")
print(f"Range : {sa['Settlement Date'].min()} -> {sa['Settlement Date'].max()}")
print(f"Mean  : ${sa['Spot Price ($/MWh)'].mean():.2f}/MWh")
print(f"Max   : ${sa['Spot Price ($/MWh)'].max():.2f}/MWh")
print("\nDONE. Now update preprocess.py glob pattern then run:")
print("  python preprocess.py")
print("  python train_models.py")