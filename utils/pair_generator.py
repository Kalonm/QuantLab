import itertools
import json
from pathlib import Path

# 1) Your full symbol list as a single string
symbols_str = """
AUDNZD AUDCAD AUDCHF AUDJPY CHFJPY EURGBP EURAUD EURJPY EURNZD EURCHF EURCAD
GBPCHF GBPJPY CADCHF CADJPY GBPAUD GBPCAD GBPNZD NZDCAD NZDCHF NZDJPY NZDUSD
USDSGD AUDSGD CHFSGD EURDKK EURHKD EURNOK EURPLN EURSEK EURSGD EURTRY EURZAR
GBPDKK GBPNOK GBPSEK GBPSGD GBPTRY NOKJPY NOKSEK SEKJPY USDCNH USDCZK USDDKK
USDHKD USDHUF USDMXN USDNOK USDPLN EURUSD GBPUSD USDCHF USDJPY USDCAD AUDUSD
XBRUSD XNGUSD XTIUSD WTI_V5 WTI_Z5 BRENT_F6 BRENT_G6 WTI_F6 Wheat_Z5 Corn_Z5
Wheat_H6 Sugar_H6 Cocoa_H6 Cotton_H6 Sbean_H6 Corn_H6 OJ_F6 Coffee_H6 XAGEUR
XAGUSD XAUEUR XAUSUD XDPUSD XPTUSD XAUAUD XAUJPY XAUCHF XAUGBP XAGAUD GCZ25
AUS200 STOXX50 F40 JP225 UK100 UUS30 US500 USTEC DE40 CHINA50 ES35 HK50 IT40
US2000 CA60 NETH35 SE30 SWI20 CHINAH SA40 NOR25 TecDE30 MidDE60 MidDE50
VIX_X5 VIX_Z4 VIX_Z5 DXY_Z5
"""

# 2) Turn into a clean Python list (dedupe + sort, in case of duplicates)
symbols = symbols_str.split()
symbols = sorted(set(symbols))

print(f"Total unique symbols: {len(symbols)}")

# 3) Generate all unique unordered pairs (combinations of 2)
pairs = []
for s1, s2 in itertools.combinations(symbols, 2):
    pairs.append({"symbol1": s1, "symbol2": s2})

print(f"Total pairs generated: {len(pairs)}")

# 4) Save to JSON file
output_path = Path("pairs.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(pairs, f, indent=2)

print(f"Saved pairs to {output_path.resolve()}")
