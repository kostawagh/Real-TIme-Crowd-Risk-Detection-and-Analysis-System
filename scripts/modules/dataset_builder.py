"""
dataset_builder.py
Builds supervised ML dataset from risk CSV files.

Input:
    data/ground_truth.csv
    output/risk/risk_umn*.csv

Output:
    output/ml_dataset.csv
"""

import os
import pandas as pd


class MLDatasetBuilder:
    def __init__(
        self,
        ground_truth_csv: str = "data/ground_truth.csv",
        risk_dir: str = "output/risk",
        output_path: str = "output/ml_dataset.csv",
    ):
        self.ground_truth_csv = ground_truth_csv
        self.risk_dir = risk_dir
        self.output_path = output_path

        self.gt_df = pd.read_csv(self.ground_truth_csv)

    # ─────────────────────────────────────────────
    # Ground-truth label helper
    # ─────────────────────────────────────────────
    @staticmethod
    def _get_label(frame: int, start: int, end: int) -> int:
        if start == -1 and end == -1:
            return 0
        return 1 if start <= frame <= end else 0

    # ─────────────────────────────────────────────
    # Main dataset builder
    # ─────────────────────────────────────────────
    def build(self) -> str:
        all_dfs = []

        for _, gt_row in self.gt_df.iterrows():

            video_name = str(gt_row["video_name"])
            start = int(gt_row["anomaly_start"])
            end = int(gt_row["anomaly_end"])

            risk_csv = os.path.join(
                self.risk_dir,
                f"risk_{video_name}.csv"
            )

            if not os.path.exists(risk_csv):
                print(f"[WARN] Missing file: {risk_csv}")
                continue

            print(f"[LOAD] {risk_csv}")

            df = pd.read_csv(risk_csv)

            # ─────────────────────────────────────
            # REMOVE HANDCRAFTED DECISION COLUMNS
            # (avoid ML learning our manual rules)
            # ─────────────────────────────────────
            drop_cols = [
                "risk",
                "risk_smooth",
                "risk_level",
            ]

            df = df.drop(
                columns=[c for c in drop_cols if c in df.columns]
            )

            # ─────────────────────────────────────
            # Add source video name
            # ─────────────────────────────────────
            df.insert(0, "video_name", video_name)

            # ─────────────────────────────────────
            # Add supervised anomaly labels
            # ─────────────────────────────────────
            df["label"] = df["frame"].apply(
                lambda f: self._get_label(
                    int(f),
                    start,
                    end
                )
            )

            all_dfs.append(df)

        # ─────────────────────────────────────────
        # Combine all videos
        # ─────────────────────────────────────────
        if not all_dfs:
            raise RuntimeError(
                "No risk CSVs were loaded. Dataset not created."
            )

        final_df = pd.concat(all_dfs, ignore_index=True)

        # ─────────────────────────────────────────
        # Save dataset
        # ─────────────────────────────────────────
        os.makedirs(
            os.path.dirname(self.output_path),
            exist_ok=True
        )

        final_df.to_csv(self.output_path, index=False)

        # ─────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("ML Dataset Created")
        print("=" * 60)

        print(f"Saved to        : {self.output_path}")
        print(f"Total rows      : {len(final_df)}")
        print(f"Normal rows     : {(final_df['label'] == 0).sum()}")
        print(f"Anomaly rows    : {(final_df['label'] == 1).sum()}")
        print(f"Videos included : {final_df['video_name'].nunique()}")

        return self.output_path


# ═══════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════
if __name__ == "__main__":

    builder = MLDatasetBuilder(
        ground_truth_csv="data/ground_truth.csv",
        risk_dir="output/risk",
        output_path="output/ml_dataset.csv",
    )

    builder.build()