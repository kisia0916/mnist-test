import numpy as np
import json
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# --- 設定値 ---
INPUT_DIM = 784
HIDDEN_1 = 128
HIDDEN_2 = 64
OUTPUT_DIM = 10
STUDY_RATE = 0.01  # 行列演算に合わせて調整
TRAIN_EPOCHS = 20

# --- 重みとバイアスの初期化 (NumPy ndarray) ---
# 乱数で初期化。np.random.randn を使うと効率的です
W1 = np.random.randn(HIDDEN_1, INPUT_DIM) * 0.01
W2 = np.random.randn(HIDDEN_2, HIDDEN_1) * 0.01
W3 = np.random.randn(OUTPUT_DIM, HIDDEN_2) * 0.01

B1 = np.zeros(HIDDEN_1)
B2 = np.zeros(HIDDEN_2)
B3 = np.zeros(OUTPUT_DIM)

def ReLU(x):
    return np.maximum(0, x)

def forward(X):
    # np.dot(重み, 入力) で各層の計算を一括処理
    z1 = np.dot(W1, X) + B1
    a1 = ReLU(z1)
    
    z2 = np.dot(W2, a1) + B2
    a2 = ReLU(z2)
    
    z3 = np.dot(W3, a2) + B3
    # 出力層は活性化関数なし（または必要に応じてSoftmax）
    return z3, a1, a2

def train(X, a1, a2, out, ans):
    global W1, W2, W3, B1, B2, B3
    
    # 1. 誤差の計算 (出力層から逆算)
    error_out = out - ans # (10,)
    
    # 2. 隠れ層2の誤差伝播 (転置行列 .T を活用)
    # ReLUの微分: a2 > 0 の部分だけ誤差を通す
    error_2 = np.dot(W3.T, error_out) * (a2 > 0)
    
    # 3. 隠れ層1の誤差伝播
    error_1 = np.dot(W2.T, error_2) * (a1 > 0)
    
    # 4. 重みとバイアスの更新 (np.outer で行列全体を一気に更新)
    # W3修正
    W3 -= STUDY_RATE * np.outer(error_out, a2)
    B3 -= STUDY_RATE * error_out
    
    # W2修正
    W2 -= STUDY_RATE * np.outer(error_2, a1)
    B2 -= STUDY_RATE * error_2
    
    # W1修正
    W1 -= STUDY_RATE * np.outer(error_1, X)
    B1 -= STUDY_RATE * error_1

def one_hot_encoding(label):
    one_hot = np.zeros(10)
    one_hot[label] = 1
    return one_hot

def save_weights(file_path="model_weights.json"):
    model_data = {
        "W1": W1.tolist(), "W2": W2.tolist(), "W3": W3.tolist(),
        "B1": B1.tolist(), "B2": B2.tolist(), "B3": B3.tolist()
    }
    with open(file_path, "w") as f:
        json.dump(model_data, f)
    print(f"モデルの重みを {file_path} に保存しました。")

def benchmark_mnist():
    global W1, W2, W3, B1, B2, B3
    
    # データ読み込み
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # 正規化とフラット化 (28x28 -> 784)
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
    
    train_acc_history = []

    print(f"MNIST BENCHMARK START (Input: {INPUT_DIM}, Nodes: {HIDDEN_1}-{HIDDEN_2})")

    for epoch in range(TRAIN_EPOCHS):
        # シャッフル
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        correct_count = 0
        
        # 学習ループ
        for s in range(len(X_train_shuffled)):
            img = X_train_shuffled[s]
            label = y_train_shuffled[s]
            
            # 推論
            out, a1, a2 = forward(img)
            
            # 正解判定
            if np.argmax(out) == label:
                correct_count += 1
            
            # 学習 (誤差逆伝播)
            ans = one_hot_encoding(label)
            train(img, a1, a2, out, ans)
            
        acc = correct_count / len(X_train)
        train_acc_history.append(acc)
        print(f"Epoch {epoch + 1}/{TRAIN_EPOCHS} - Accuracy: {acc*100:.2f}%")

    # 最終評価
    test_correct = 0
    for s in range(len(X_test)):
        out, _, _ = forward(X_test[s])
        if np.argmax(out) == y_test[s]:
            test_correct += 1
    
    print(f"\nFINAL TEST ACCURACY: {test_correct / len(X_test) * 100:.2f}%")
    save_weights()

if __name__ == "__main__":
    benchmark_mnist()