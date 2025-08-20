import pandas as pd
import argparse
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def find_audio_paths(base_dir):   # 掃所有層級的 .wav
    audio_dict = {}
    for path in Path(base_dir).rglob("*.wav"):
        audio_dict[path.name] = str(path)
    return audio_dict

def preprocess_csv_and_audio(raw_data_dir, output_csv_path, test_ratio=0.2, random_seed=42):
    csv_paths = list(Path(raw_data_dir).glob("*.csv"))
    if not csv_paths:
        logging.error(f"No CSV files found in: {raw_data_dir}")
        return

    logging.info(f"Found {len(csv_paths)} CSV files.")
    dfs = []
    for csv in csv_paths:
        df = pd.read_csv(csv)
        df["source_csv"] = csv.name   # 紀錄來源檔名
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    audio_map = find_audio_paths(raw_data_dir)
    logging.info(f"Indexed {len(audio_map)} audio files.")

    processed_rows = []
    no_note_count = 0
    with_note_count = 0
    seen_files = set()
    duplicate_in_rows = 0

    required_cols = ["檔名", "source_csv", "客語漢字", "客語拼音", "備註"]
    missing_cols = [c for c in ["檔名", "客語漢字", "客語拼音"] if c not in all_df.columns]
    if missing_cols:
        logging.error(f"Missing required columns in CSVs: {missing_cols}")
        return

    for _, row in all_df.iterrows():
        file_name = row["檔名"]
        audio_path = audio_map.get(file_name)

        # 檢查重複
        if file_name in seen_files:
            duplicate_in_rows += 1
        else:
            seen_files.add(file_name)

        if audio_path is None:
            logging.warning(f"Audio file not found: {file_name} (from {row.get('source_csv', pd.NA)})")
            continue

        # 需有客語漢字與拼音
        if pd.isna(row["客語漢字"]) or pd.isna(row["客語拼音"]):
            continue

        # 統計有備註
        if pd.isna(row["備註"]) or str(row["備註"]).strip() == "":
            no_note_count += 1
            processed_rows.append({
                "audio_path": audio_path,
                "客語漢字": row["客語漢字"],
                "客語拼音": row["客語拼音"],
                # "備註": row["備註"],
                # "source_csv": row.get("source_csv", pd.NA)
            })
        else:
            with_note_count += 1

    if not processed_rows:
        logging.warning("No valid data found.")
        logging.info(f"Items with empty note: {no_note_count}, with note (discarded): {with_note_count}")
        logging.info(f"Duplicate '檔名' found in processed rows: {duplicate_in_rows}")
        return

    df_output = pd.DataFrame(processed_rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df_output.to_csv(output_csv_path, index=False)

    # train/test split
    n = len(df_output)
    rng = np.random.default_rng(seed=random_seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    test_size = int(round(n * float(test_ratio)))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    df_train = df_output.iloc[train_idx].reset_index(drop=True)
    df_test = df_output.iloc[test_idx].reset_index(drop=True)

    # 輸出兩個額外 CSV
    stem = output_csv_path.stem
    suffix = output_csv_path.suffix or ".csv"
    train_path = output_csv_path.with_name(f"{stem}_train{suffix}")
    test_path = output_csv_path.with_name(f"{stem}_test{suffix}")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

    logging.info(f"Saved {len(df_output)} entries to {output_csv_path}")
    logging.info(f"Train/Test split with test_ratio={test_ratio} (seed={random_seed}) -> "
                 f"train: {len(df_train)}, test: {len(df_test)}")
    logging.info(f"Items with empty note (kept): {no_note_count}, with note (discarded): {with_note_count}")
    logging.info(f"Duplicate '檔名' found in processed rows: {duplicate_in_rows}")
    logging.info(f"Train CSV: {train_path}")
    logging.info(f"Test  CSV: {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hakka CSV + audio files to simplified CSV format with train/test split.")
    parser.add_argument("--raw_data_dir", type=str, default="data/FSR-2025-Hakka-train", help="Directory containing CSVs and audio folders")
    parser.add_argument("--output_csv", type=str, default="data/hakka_data.csv", help="Path to save the filtered full CSV (train/test will be derived)")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Proportion of data to use for test split (e.g., 0.2)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducible split")

    args = parser.parse_args()

    preprocess_csv_and_audio(
        raw_data_dir=args.raw_data_dir,
        output_csv_path=args.output_csv,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )
