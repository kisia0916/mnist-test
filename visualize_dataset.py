import pandas as pd
import matplotlib.pyplot as plt
import random


def infer_side_length(feature_count):
    side = int(feature_count**0.5)
    if side * side != feature_count:
        return None
    return side

def visualize_dataset(csv_file, num_to_show=25):
    """
    保存されたCSVデータを読み込み、タイル状に並べて表示する
    """
    try:
        # CSVファイルを読み込む（ヘッダーなし）
        df = pd.read_csv(csv_file, header=None)
        print(f"'{csv_file}' を読み込みました。データ数: {len(df)}")
    except FileNotFoundError:
        print(f"エラー: '{csv_file}' が見つかりません。先にデータ生成スクリプトを実行してください。")
        return

    if len(df) == 0:
        print("エラー: データが空です。")
        return

    feature_count = df.shape[1] - 1
    side = infer_side_length(feature_count)
    if side is None:
        print(f"エラー: 特徴量数 {feature_count} は正方画像に変換できません。")
        return

    # 全データからランダムに抽出
    if len(df) < num_to_show:
        num_to_show = len(df)
    indices = random.sample(range(len(df)), num_to_show)
    sample_data = df.iloc[indices]

    # タイル状に並べる設定（例：25枚なら 5x5）
    cols = int(num_to_show**0.5)
    rows = (num_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5), squeeze=False)
    fig.suptitle(f"Random Samples from {csv_file}", fontsize=16)

    # 抽出したデータを1枚ずつ描画
    for i, (index, row) in enumerate(sample_data.iterrows()):
        ax = axes[i // cols, i % cols]
        
        # 特徴量を取り出して side x side に整形
        pixels = row.iloc[:-1].values.reshape(side, side)
        # 最後の列をラベル（正解）として取得
        label = int(row.iloc[-1])
        
        # 描画（白黒反転 'gray_r' で手書き風に）
        ax.imshow(pixels, cmap='gray_r', interpolation='nearest')
        ax.set_title(f"Label: {label}", fontsize=10)
        ax.axis('off') # 軸（目盛り）を非表示

    # 残りの空白マスを非表示にする
    for i in range(num_to_show, rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # タイトルと重ならないように調整
    plt.show()

# --- 実行 ---
# 先ほどのスクリプトで生成したファイル名を指定
visualize_dataset("digits_dataset.csv", num_to_show=36) # 36枚 (6x6) 表示