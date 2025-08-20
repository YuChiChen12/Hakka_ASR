import argparse
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def transform_csv(in_csv: Path, out_csv: Path, sep: str = " || ", overwrite: bool = False):
    df = pd.read_csv(in_csv)
    required = ["audio_path", "客語漢字", "客語拼音"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{in_csv} 缺少必要欄位：{col}")

    df["audio_path"] = df["audio_path"].astype(str).str.strip()
    df["客語漢字"] = df["客語漢字"].fillna("").astype(str).str.strip()
    df["客語拼音"] = df["客語拼音"].fillna("").astype(str).str.strip()
    df["拼音漢字"] = df["客語拼音"] + sep + df["客語漢字"]
    df["漢字拼音"] = df["客語漢字"] + sep + df["客語拼音"]

    if overwrite:
        df_out = df  # 保留所有欄位
    else:
        df_out = df[["audio_path", "拼音漢字", "漢字拼音"]]  # 只留三個欄位

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info(f"轉檔完成：{in_csv} → {out_csv}（{len(df_out)} 筆）")

def make_out_path(in_csv: Path, prefix: str, suffix: str, ext: str, overwrite: bool) -> Path:
    if overwrite:
        return in_csv  # 直接覆寫原檔
    use_ext = ext if ext else in_csv.suffix
    filename = f"{prefix}{in_csv.stem}{suffix}{use_ext}"
    return in_csv.with_name(filename)

def main():
    parser = argparse.ArgumentParser(description="從 Hakka CSV 產生 label 欄位（拼音+漢字 / 漢字+拼音）")
    parser.add_argument("--input_glob", type=str, required=True,
                        help="輸入 CSV 的 glob 模式，如 'data/*.csv' 或 'data/**/*.csv'")
    parser.add_argument("--sep", type=str, default=" || ",
                        help="兩段文字之間的分隔符，預設 ' || '")
    parser.add_argument("--out_prefix", type=str, default="",
                        help="輸出檔名前綴（預設空字串）")
    parser.add_argument("--out_suffix", type=str, default="_labels",
                        help="輸出檔名後綴（預設 _labels）")
    parser.add_argument("--out_ext", type=str, default="",
                        help="輸出副檔名（含點，如 '.csv'）。預設空字串=沿用原始副檔名")
    parser.add_argument("--overwrite", action="store_true",
                        help="啟用後會直接覆寫原 CSV 檔（保留其他欄位）")
    args = parser.parse_args()

    input_paths = [Path(p) for p in sorted(Path().glob(args.input_glob))]
    if not input_paths:
        logging.error(f"找不到符合的檔案：{args.input_glob}")
        return

    for in_csv in input_paths:
        try:
            out_csv = make_out_path(
                in_csv=in_csv,
                prefix=args.out_prefix,
                suffix=args.out_suffix,
                ext=args.out_ext,
                overwrite=args.overwrite
            )
            transform_csv(in_csv, out_csv, sep=args.sep, overwrite=args.overwrite)
        except Exception as e:
            logging.error(f"處理失敗：{in_csv}，錯誤：{e}")

if __name__ == "__main__":
    main()
