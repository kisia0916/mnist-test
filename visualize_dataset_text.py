import argparse
import random

import pandas as pd

BLOCK_LEVELS = " ░▒▓█"


def infer_side_length(feature_count: int):
    side = int(feature_count**0.5)
    if side * side != feature_count:
        return None
    return side


def pixel_to_block(value: float) -> str:
    value = max(0.0, min(1.0, float(value)))
    index = int(round(value * (len(BLOCK_LEVELS) - 1)))
    return BLOCK_LEVELS[index]


def render_block_digit(pixels) -> str:
    lines = []
    for row in pixels:
        # 横幅を2文字にして見た目の比率を整える
        line = "".join(pixel_to_block(v) * 2 for v in row)
        lines.append(line)
    return "\n".join(lines)


def print_dataset_text(csv_file: str, num_to_show: int = 10, random_pick: bool = True) -> None:
    try:
        df = pd.read_csv(csv_file, header=None)
        print(f"'{csv_file}' を読み込みました。データ数: {len(df)}")
    except FileNotFoundError:
        print(f"エラー: '{csv_file}' が見つかりません。")
        return

    if len(df) == 0:
        print("データが空です。")
        return

    if df.shape[1] < 2:
        print("エラー: CSVの列数が不足しています。")
        return

    feature_count = df.shape[1] - 1
    side = infer_side_length(feature_count)
    if side is None:
        print(f"エラー: 特徴量数 {feature_count} は正方画像に変換できません。")
        return

    num_to_show = max(1, min(num_to_show, len(df)))

    if random_pick:
        indices = random.sample(range(len(df)), num_to_show)
        sample_data = df.iloc[indices]
    else:
        sample_data = df.iloc[:num_to_show]

    for i, (index, row) in enumerate(sample_data.iterrows(), start=1):
        pixels = row.iloc[:-1].values.reshape(side, side)
        label = int(row.iloc[-1])

        print(f"\n=== Sample {i}/{num_to_show} (row={index}, label={label}) ===")
        print(render_block_digit(pixels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Digits dataset text visualizer")
    parser.add_argument("--file", default="digits_dataset_drawn.csv", help="CSV file path")
    parser.add_argument("--num", type=int, default=10, help="Number of samples to show")
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Show first N rows in order instead of random sampling",
    )
    args = parser.parse_args()

    print_dataset_text(args.file, num_to_show=args.num, random_pick=not args.ordered)


if __name__ == "__main__":
    main()
