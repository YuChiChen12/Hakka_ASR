import pandas as pd
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def find_audio_paths(base_dir):
    audio_dict = {}
    for path in Path(base_dir).rglob("*/*.wav"):
        audio_dict[path.name] = str(path)
    for path in Path(base_dir).rglob("*/*/*.wav"):
        audio_dict[path.name] = str(path)
    return audio_dict

def preprocess_csv_and_audio(raw_data_dir, output_csv_path):
    csv_paths = list(Path(raw_data_dir).glob("*.csv"))
    if not csv_paths:
        logging.error(f"No CSV files found in: {raw_data_dir}")
        return

    logging.info(f"Found {len(csv_paths)} CSV files.")
    dfs = []
    for csv in csv_paths:
        df = pd.read_csv(csv)
        df["source_csv"] = str(csv)   # 新增來源欄位
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    audio_map = find_audio_paths(raw_data_dir)
    logging.info(f"Indexed {len(audio_map)} audio files.")

    processed_rows = []
    no_note_count = 0
    with_note_count = 0
    seen_files = set()
    duplicate_in_rows = 0

    for _, row in all_df.iterrows():
        file_name = row["檔名"]
        audio_path = audio_map.get(file_name)

        # 檢查重複
        if file_name in seen_files:
            duplicate_in_rows += 1
        else:
            seen_files.add(file_name)

        if audio_path is None:
            logging.warning(f"Audio file not found: {file_name} (from {row['source_csv']})")
            continue

        if pd.isna(row["客語漢字"]) or pd.isna(row["客語拼音"]):
            continue

        if pd.isna(row["備註"]):
            no_note_count += 1
            
        else:
            with_note_count += 1
        
        processed_rows.append({
            "audio_path": audio_path,
            "客語漢字": row["客語漢字"],
            "客語拼音": row["客語拼音"],
            "備註": row["備註"],
        })

    if not processed_rows:
        logging.warning("No valid data found.")
        return

    df_output = pd.DataFrame(processed_rows)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv_path, index=False)

    logging.info(f"Saved {len(df_output)} entries to {output_csv_path}")
    logging.info(f"Items with empty note: {no_note_count}, with note: {with_note_count}")
    logging.info(f"Duplicate '檔名' found in processed rows: {duplicate_in_rows}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Hakka CSV + audio files to simplified CSV format.")
    parser.add_argument("--raw_data_dir", type=str, default="data/FSR-2025-Hakka-train", help="Directory containing CSVs and audio folders")
    parser.add_argument("--output_csv", type=str, default="data/hakka_data.csv", help="Path to save the output CSV")

    args = parser.parse_args()

    preprocess_csv_and_audio(
        raw_data_dir=args.raw_data_dir,
        output_csv_path=args.output_csv
    )
