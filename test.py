# 自動で6万枚の学習データと1万枚のテストデータに分かれてダウンロードされる
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 28x28 -> 784 の1次元ベクトルに変換
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

print(f"学習データの数: {len(X_train)}")
print(y_train[0])
# print(f"X_train shape: {X_train.shape}")