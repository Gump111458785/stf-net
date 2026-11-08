import os
import json
import sys

def format_row(cols, widths):
    return "| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(cols, widths)) + " |"

def format_separator(widths):
    return "|-" + "-|-".join("-" * w for w in widths) + "-|"

def main():
    ref_path = "results/paper_reference_results.json"
    if not os.path.exists(ref_path):
        print(f"Error: {ref_path} not found.")
        sys.exit(1)

    with open(ref_path, "r") as f:
        ref_data = json.load(f)

    print("=" * 80)
    print("STF-Net Reproduction Verification Audit")
    print("=" * 80)

    headers = ["Dataset", "Horizon", "Model", "Ref MSE", "Ref MAE"]
    widths = [12, 10, 12, 12, 12]
    print(format_row(headers, widths))
    print(format_separator(widths))

    for ds in ["ETTh1", "Weather", "Electricity"]:
        for hz in ["96", "192", "336", "720"]:
            for m in ["STF-Net", "DLinear"]:
                metrics = ref_data[ds][hz][m]
                row = [ds, hz, m, f"{metrics['MSE']:.3f}", f"{metrics['MAE']:.3f}"]
                print(format_row(row, widths))

    print("\nDataset Integrity Audit Check:")
    print("Checking benchmark dataset files in ./data ...")
    missing_files = []
    for ds in ["ETTh1.csv", "Weather.csv", "Electricity.csv"]:
        target = os.path.join("data", ds)
        if not os.path.exists(target):
            missing_files.append(target)
            print(f"  [MISSING] {target}")
        else:
            print(f"  [EXISTS]  {target}")

    if missing_files:
        print("\n[AUDIT RESULT: BLOCKED]")
        print("Required benchmark datasets are missing! Dataset download failed during pipeline initialization.")
        print("Please check scripts/download_data.sh network mirror endpoints.")
    else:
        print("\n[AUDIT RESULT: READY]")

if __name__ == "__main__":
    main()
