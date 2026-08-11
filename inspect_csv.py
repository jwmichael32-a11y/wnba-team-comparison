import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
print(f"{path}: {len(df)} rows")
blank_rows = df[df.isna().all(axis=1)]
if not blank_rows.empty:
    print(f"  {len(blank_rows)} fully-blank row(s) at index: {blank_rows.index.tolist()}")
partial = df[df[['Season', 'team_code']].isna().any(axis=1)]
if not partial.empty:
    print("  Row(s) with a NaN Season or team_code specifically:")
    print(partial)
