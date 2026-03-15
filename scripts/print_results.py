"""Print evaluation results from a JSON file."""
import json
import argparse

AUGS = ["time_stretch", "pitch_shift", "reverberation", "noise"]
AUG_DISPLAY = {
    "time_stretch": "Time Stretch",
    "pitch_shift": "Pitch Shift",
    "reverberation": "Reverberation",
    "noise": "Noise",
}


def print_ued(ued: dict):
    baseline = ued.get("_baseline")
    if baseline:
        print(f"  {'Augmentation':<16}  {'E0 mean':>8}  {'±SEM':>6}  {'E1 mean':>8}  {'±SEM':>6}  {'Δ':>7}")
        print("  " + "-" * 62)
        for aug in AUGS:
            b = baseline.get(aug, {})
            e = ued.get(aug, {})
            b_str = f"{b['mean_ued']:8.2f}  {b['sem_ued']:6.2f}" if b else f"{'N/A':>8}  {'':6}"
            e_str = f"{e['mean_ued']:8.2f}  {e['sem_ued']:6.2f}" if e else f"{'N/A':>8}  {'':6}"
            delta = f"{e['mean_ued'] - b['mean_ued']:+7.2f}" if b and e else ""
            print(f"  {AUG_DISPLAY[aug]:<16}  {b_str}  {e_str}  {delta}")
    else:
        print(f"  {'Augmentation':<16}  {'E1 mean':>8}  {'±SEM':>6}  {'n':>5}")
        print("  " + "-" * 42)
        for aug in AUGS:
            e = ued.get(aug, {})
            if e:
                print(f"  {AUG_DISPLAY[aug]:<16}  {e['mean_ued']:8.2f}  {e['sem_ued']:6.2f}  {e['n']:5}")


def main():
    parser = argparse.ArgumentParser(description="Print evaluation results from JSON")
    parser.add_argument("files", nargs="+", help="Result JSON files to display")
    args = parser.parse_args()

    for path in args.files:
        with open(path) as f:
            data = json.load(f)

        print(f"\n{'='*64}")
        print(f"  {path}")
        print(f"{'='*64}")

        if "ued" in data:
            print("\n  UED (Unit Edit Distance) — lower is better\n")
            print_ued(data["ued"])

        if "abx" in data:
            abx = data["abx"]
            print("\n  ABX Error Rate — lower is better\n")
            print(f"  {'Setup':<24}  {'Error (%)':>10}  {'n':>6}")
            print("  " + "-" * 44)
            for key, val in abx.items():
                if isinstance(val, dict):
                    err = val.get("error_rate", val.get("abx_error", "N/A"))
                    n = val.get("n", "")
                    print(f"  {key:<24}  {err:10.2f}  {n:6}")

        print()


if __name__ == "__main__":
    main()
