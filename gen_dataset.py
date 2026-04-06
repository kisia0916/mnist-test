import numpy as np
import pandas as pd
import random

def generate_handwritten_data(samples_per_digit=100):
    dataset = []
    
    # 各数字の基本骨格 (8x8)
    base_patterns = {
        0: [18, 19, 20, 21, 25, 30, 33, 38, 41, 46, 49, 54, 58, 59, 60, 61],
        1: [3, 11, 19, 27, 35, 43, 51, 59],
        2: [10, 11, 12, 13, 22, 29, 36, 43, 50, 51, 52, 53, 54],
        3: [10, 11, 12, 13, 22, 28, 29, 30, 38, 46, 50, 51, 52, 53],
        4: [1, 9, 17, 25, 26, 27, 28, 29, 30, 22, 38, 46, 54],
        5: [10, 11, 12, 13, 18, 26, 27, 28, 29, 38, 46, 50, 51, 52, 53],
        6: [18, 19, 20, 21, 25, 33, 34, 35, 36, 37, 41, 46, 49, 54, 58, 59, 60, 61],
        7: [1, 2, 3, 4, 5, 6, 14, 21, 28, 35, 42, 49],
        8: [18, 19, 20, 21, 25, 30, 34, 35, 36, 37, 41, 46, 49, 54, 58, 59, 60, 61],
        9: [18, 19, 20, 21, 25, 30, 34, 35, 36, 37, 38, 46, 54]
    }

    for digit in range(10):
        base = base_patterns[digit]
        for _ in range(samples_per_digit):
            # 64マスの真っ白なキャンバス
            img = np.zeros(64)
            
            # 基本骨格に値をセット
            for pos in base:
                if 0 <= pos < 64:
                    # 筆圧のばらつき (0.7〜1.0)
                    img[pos] = random.uniform(0.7, 1.0)
            
            # 手書きらしいノイズと「揺れ」を追加
            # 1. 全体に薄いノイズ
            img += np.random.normal(0, 0.05, 64)
            # 2. ランダムな位置にドットを追加
            for _ in range(3):
                img[random.randint(0, 63)] += random.uniform(0, 0.3)
            
            # 0〜1の範囲にクリップ
            img = np.clip(img, 0, 1)
            
            # 最後にラベル（正解）を結合
            data_row = np.append(img, digit)
            dataset.append(data_row)
            
    # シャッフルして保存
    random.shuffle(dataset)
    df = pd.DataFrame(dataset)
    df.to_csv("digits_dataset.csv", index=False, header=False)
    print(f"保存完了: {len(dataset)} 件のデータを 'digits_dataset.csv' に書き出しました。")

generate_handwritten_data(100)