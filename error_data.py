import argparse
import pandas as pd
import logging
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Filter rows where reference != prediction and rename output column by track")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_csv", type=str, default="mismatched.csv", help="Path to save the filtered CSV (default: mismatched.csv)")
    parser.add_argument("--track", type=str, choices=["track1", "track2"], required=True,
                        help="track1 => rename to '客語漢字'; track2 => rename to '客語拼音'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"Reading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df.columns = df.columns.str.strip()

    for col in ["audio_path", "reference", "prediction"]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}'. Found columns: {list(df.columns)}")

    logger.info("Filtering rows where reference != prediction...")
    df_diff = df[df["reference"] != df["prediction"]]

    out_col = "客語漢字" if args.track == "track1" else "客語拼音"
    out_df = df_diff[["audio_path", "reference"]].rename(columns={"reference": out_col})

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    logger.info(f"Total rows in original file: {len(df)}")
    logger.info(f"Rows with mismatches: {len(df_diff)}")
    logger.info(f"Saved columns: {list(out_df.columns)}")
    logger.info(f"Filtered CSV saved to: {output_path.resolve()}")

if __name__ == "__main__":
    main()
