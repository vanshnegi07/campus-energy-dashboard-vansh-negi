"""
gradebook.py
Simple GradeBook Analyzer CLI
Name: Vansh Negi
Course: B.Tech CSE AI & ML - 1st Year
Section: A
Enrollment Number: 2501730158
Subject: Programming for Problem Solving using Python
# I made this for Lab Assignment 5(Capstone) 
"""



# main.py
# Clean, robust campus energy pipeline (student-friendly)
# Handles messy CSVs like ",timestamp,kwh,,,," or lines with extra commas
# Requires: pandas, matplotlib
# Run: python3 main.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ----------------------------
# find header line with timestamp & kwh tokens
# ----------------------------
def find_header_line(lines):
    seps = [",", ";", "\t"]
    for idx, ln in enumerate(lines[:80]):  # scan first 80 lines
        if not ln or ln.strip() == "":
            continue
        low = ln.lower()
        for sep in seps:
            parts = [p.strip().lower() for p in ln.split(sep)]
            if any("timestamp" in p or "time" == p or "date" in p for p in parts) and \
               any("kwh" in p or "energy" in p or "value" in p or "consump" in p for p in parts):
                return idx, sep, [p.strip() for p in ln.split(sep)]
    return None, None, None

# ----------------------------
# Robust file parser 
# ----------------------------
def parse_file(path: Path):
    """
    Find header line anywhere, detect sep, then extract timestamp and kwh columns
    even if header/rows have leading commas or extra empty tokens.
    Returns DataFrame with columns ['timestamp','kwh'] or None.
    """
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    if not text.strip():
        return None

    lines = text.splitlines()
    # remove lines that are only separators or whitespace for header search convenience
    nonempty = [ln for ln in lines if ln.strip() != ""]
    if not nonempty:
        return None

    header_idx, sep, header_parts = find_header_line(lines)
    if header_idx is None:
        # no explicit header; try a quick fallback: assume first non-empty line is header
        for i, ln in enumerate(lines[:10]):
            if ln.strip() and ("," in ln or ";" in ln or "\t" in ln):
                header_idx = i
                sep = "," if "," in ln else (";" if ";" in ln else "\t")
                header_parts = [p.strip() for p in ln.split(sep)]
                # only accept if header_parts contain time-like and kwh-like tokens
                low = [p.lower() for p in header_parts]
                if not (any("timestamp" in p or "time" == p or "date" in p for p in low) and
                        any("kwh" in p or "energy" in p or "value" in p for p in low)):
                    header_idx = None
                else:
                    break
    if header_idx is None:
        # give up early
        return None

    # find indexes of timestamp and kwh
    t_idx = None
    k_idx = None
    for i, token in enumerate(header_parts):
        tok_low = str(token).lower()
        if t_idx is None and ("timestamp" in tok_low or "time" == tok_low or "date" in tok_low):
            t_idx = i
        if k_idx is None and ("kwh" in tok_low or "energy" in tok_low or "value" in tok_low or "consump" in tok_low):
            k_idx = i

    if t_idx is None or k_idx is None:
        return None

    # collect rows after header_idx
    rows = []
    for ln in lines[header_idx + 1:]:
        if not ln or ln.strip() == "":
            continue
        # skip lines that are only separators
        if set(ln.strip()).issubset({",", ";", "\t", " "}):
            continue
        parts = [p for p in ln.split(sep)]
        # pad short rows
        if max(t_idx, k_idx) >= len(parts):
            parts += [""] * (max(t_idx, k_idx) - len(parts) + 1)
        t_tok = parts[t_idx].strip()
        k_tok = parts[k_idx].strip()
        # if tokens empty try collapsing repeated separators (handles ", ,timestamp,kwh")
        if (t_tok == "" or k_tok == "") and sep == ",":
            compact = ln.replace(",,", ",")
            parts2 = [p for p in compact.split(sep)]
            if max(t_idx, k_idx) < len(parts2):
                t_tok = parts2[t_idx].strip()
                k_tok = parts2[k_idx].strip()
        if t_tok == "" or k_tok == "":
            continue
        rows.append((t_tok, k_tok))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["timestamp", "kwh"])
    # clean tokens
    df["timestamp"] = df["timestamp"].str.strip().str.strip('"').str.strip("'")
    df["kwh"] = df["kwh"].str.strip().str.strip('"').str.strip("'").str.replace(",", "")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "kwh"])
    if df.empty:
        return None
    df = df.reset_index(drop=True)
    return df[["timestamp", "kwh"]]

# ----------------------------
# Simple classes and helpers
# ----------------------------
class MeterReading:
    def __init__(self, t, k):
        self.t = pd.to_datetime(t)
        self.k = float(k)

class Building:
    def __init__(self, name):
        self.name = name
        self.reads = []

    def add(self, r):
        self.reads.append(r)

    def summary(self):
        vals = [r.k for r in self.reads]
        if not vals:
            return {"Total":0, "Min":0, "Max":0, "Average":0}
        return {"Total":sum(vals), "Min":min(vals), "Max":max(vals), "Average":sum(vals)/len(vals)}

def daily_totals(df):
    return df.set_index("timestamp")["kwh"].resample("D").sum()

# ----------------------------
# Plotting & outputs
# ----------------------------
def create_dashboard(df, out_png):
    if df.empty:
        print("No data to plot.")
        return
    daily = daily_totals(df)
    avg_by_build = df.groupby("Building")["kwh"].mean()
    try:
        peaks = df.loc[df.groupby("Building")["kwh"].idxmax()].dropna(subset=["timestamp"])
    except Exception:
        peaks = pd.DataFrame(columns=df.columns)

    fig, axs = plt.subplots(3,1, figsize=(10,12))
    axs[0].plot(daily.index, daily.values, marker="o")
    axs[0].set_title("Daily Campus Consumption")
    axs[1].bar(avg_by_build.index, avg_by_build.values)
    axs[1].set_title("Average kWh per Building")
    axs[1].tick_params(axis='x', rotation=45)
    if not peaks.empty:
        axs[2].scatter(peaks["timestamp"], peaks["kwh"])
        for _, r in peaks.iterrows():
            axs[2].annotate(r["Building"], (r["timestamp"], r["kwh"]), textcoords="offset points", xytext=(5,5), fontsize=8)
    axs[2].set_title("Peak readings per Building")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print("Saved dashboard ->", out_png)

# ----------------------------
# Main
# ----------------------------
def main():
    print("Starting final parser pipeline...\n")
    files = sorted(DATA_DIR.glob("*.csv"))
    if not files:
        print("No CSVs in data/. Place your CSV files there and re-run.")
        return

    frames = []
    for f in files:
        print("Processing:", f.name)
        df = parse_file(f)
        if df is None:
            print("  -> Could not parse", f.name)
            preview = "\n".join(f.read_text(errors="ignore").splitlines()[:12])
            print("  File preview (first lines):")
            print(preview)
            continue
        df["Building"] = f.stem
        print("  -> Parsed OK (rows):", len(df))
        frames.append(df)

    if not frames:
        print("No valid CSVs parsed. Fix inputs and run again.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined["kwh"] = pd.to_numeric(combined["kwh"], errors="coerce").fillna(0)

    # save cleaned file
    cleaned_path = OUTPUT_DIR / "cleaned_energy_data.csv"
    combined.to_csv(cleaned_path, index=False)
    print("\nSaved cleaned_energy_data.csv ->", cleaned_path)

    # build manager and summary CSV
    mgr = {}
    for _, row in combined.iterrows():
        b = row["Building"]
        if b not in mgr:
            mgr[b] = Building(b)
        mgr[b].add(MeterReading(row["timestamp"], row["kwh"]))

    rows = []
    for name in sorted(mgr.keys()):
        s = mgr[name].summary()
        rows.append({"Building":name, **s})
    summary_csv = OUTPUT_DIR / "building_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print("Saved building_summary.csv ->", summary_csv)

    # text summary
    total = combined["kwh"].sum()
    highest = combined.groupby("Building")["kwh"].sum().idxmax()
    summary_txt = OUTPUT_DIR / "summary.txt"
    summary_txt.write_text(f"Total campus consumption (kWh): {total:.2f}\nHighest consuming building: {highest}\n")
    print("Saved summary.txt ->", summary_txt)

    # dashboard
    create_dashboard(combined, OUTPUT_DIR / "dashboard.png")
    print("\nAll done. Check the 'output/' folder.")

if __name__ == "__main__":
    main()
