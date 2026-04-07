import random
#dataset読み込み用
import numpy as np
import json

#ベンチマーク用
from tensorflow.keras.datasets import mnist


GRID_NUM = 784
HIDDEN_LAYER_NUM = 2
HIDDEN_LAYER_NODE_NUM = 16
FIRST_LAYER_DIMENSTION = 784
NORM_LAYER_DIMENSTION = 16
STUDY_RATE = 0.001
TRAIN_EPOCHS = 50
ans_weight = []
hidden_layer = []
biases = []
def gen_random_weight():
    #set first weight of hidden_layer
    for i in range(HIDDEN_LAYER_NUM):
        layer_weights = []
        for _ in range(HIDDEN_LAYER_NODE_NUM):
            layer_weights.append([random.uniform(-0.1,0.1) for _ in range(FIRST_LAYER_DIMENSTION if i == 0 else NORM_LAYER_DIMENSTION)])
        hidden_layer.append(layer_weights)
    #set first weight of ans
    for _ in range(10):
        ans_weight.append([random.uniform(-0.1,0.1) for _ in range(HIDDEN_LAYER_NODE_NUM)])
def gen_random_bias():
    for i in range(HIDDEN_LAYER_NUM+1):
        biases.append([0]*HIDDEN_LAYER_NODE_NUM if i < HIDDEN_LAYER_NUM else [0]*10)
def calc_dot(vec1,vec2):
    if len(vec1) != len(vec2):
        quit()
    dot = 0
    for i in range(len(vec1)):
        dot+=vec1[i]*vec2[i]
    return float(dot)
def ReLU(vec):
    result = []
    for i in vec:
        result.append(i if i > 0 else 0)
    return result


def forward(first_input):
    now_vec = first_input.copy()
    layer_1_result = []
    layer_2_result = []
    for i in range(HIDDEN_LAYER_NUM):
        new_vec = []
        for s in range(HIDDEN_LAYER_NODE_NUM):
            new_vec.append(calc_dot(now_vec,hidden_layer[i][s])+biases[i][s])
        now_vec = ReLU(new_vec)
        if i == 0:
            layer_1_result = now_vec.copy()
        elif i == 1:
            layer_2_result = now_vec.copy()
    result = []
    for s in range(len(ans_weight)):
        result.append(calc_dot(now_vec,ans_weight[s])+biases[-1][s])
    return result,layer_1_result,layer_2_result

def train(first_input,layer_1_result,layer_2_result,out_put,ans):
    out_put_error = [out_put[i]-ans[i] for i in range(len(ans))]

    layer_2_error = []
    for i in range(NORM_LAYER_DIMENSTION):
        transposed = []
        for s in range(10):
            transposed.append(ans_weight[s][i])
        if layer_2_result[i] != 0:
            layer_2_error.append(calc_dot(transposed,out_put_error))
        else:
            layer_2_error.append(0)

    layer_1_error = []
    for i in range(HIDDEN_LAYER_NODE_NUM):
        transposed = []
        for s in range(NORM_LAYER_DIMENSTION):
            transposed.append(hidden_layer[1][s][i])
        if layer_1_result[i] != 0:
            layer_1_error.append(calc_dot(transposed,layer_2_error))
        else:
            layer_1_error.append(0)

    for i in range(10):
        biases[-1][i] = biases[-1][i] - (STUDY_RATE*out_put_error[i])
        for s in range(NORM_LAYER_DIMENSTION):
            ans_weight[i][s] = ans_weight[i][s] - (STUDY_RATE*out_put_error[i]*layer_2_result[s])
    for i in range(HIDDEN_LAYER_NODE_NUM):
        biases[1][i] = biases[1][i] - (STUDY_RATE*layer_2_error[i])
        for s in range(NORM_LAYER_DIMENSTION):
            hidden_layer[1][i][s] = hidden_layer[1][i][s] - (STUDY_RATE*layer_2_error[i]*layer_1_result[s])
    for i in range(HIDDEN_LAYER_NODE_NUM):
        biases[0][i] = biases[0][i] - (STUDY_RATE*layer_1_error[i])
        for s in range(FIRST_LAYER_DIMENSTION):
            hidden_layer[0][i][s] = hidden_layer[0][i][s] - (STUDY_RATE*layer_1_error[i]*first_input[s])

def one_hot_encoding(label):
    one_hot = [0]*10
    one_hot[label] = 1
    return one_hot

##################################################################################################################
def load_digits_data(file_path):
    try:
        raw_data = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        print(f"ファイルを読み込みました: {file_path}")
    except FileNotFoundError:
        print("エラー: 指定されたファイルが見つかりません。")
        return None, None
    except ValueError as e:
        print(f"エラー: CSVの読み込みに失敗しました: {e}")
        return None, None

    raw_data = np.atleast_2d(raw_data)
    if raw_data.shape[1] < 2:
        print("エラー: CSVの列数が不足しています。")
        return None, None

    feature_count = raw_data.shape[1] - 1
    if feature_count != FIRST_LAYER_DIMENSTION:
        print(
            f"エラー: 特徴量次元が不一致です。期待値={FIRST_LAYER_DIMENSTION}, 実データ={feature_count}"
        )
        return None, None

    X = raw_data[:, :-1]
    y = raw_data[:, -1].astype(np.int32)
    return X, y



def save_weights(file_path="model_weights.json"):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert(item) for item in obj]
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    model_data = {
        "hidden_layer": convert(hidden_layer),
        "ans_weight": convert(ans_weight),
        "biases": convert(biases)
    }
    
    with open(file_path, "w") as f:
        json.dump(model_data, f)
    
    print(f"モデルの重みを {file_path} に保存しました。")

def load_weights(file_path="model_weights.json"):
    global hidden_layer, ans_weight, biases
    try:
        with open(file_path, "r") as f:
            model_data = json.load(f)
            hidden_layer = model_data["hidden_layer"]
            ans_weight = model_data["ans_weight"]
            biases = model_data["biases"]
        print(f"モデルの重みを {file_path} から読み込みました。")
    except FileNotFoundError:
        print("保存されたモデルが見つかりません")
        quit()

def run_1():        
    gen_random_weight()
    gen_random_bias()
    X, y = load_digits_data("digits_dataset.csv")
    for i in range(TRAIN_EPOCHS):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        correct_count = 0
        for s in range(len(X)):
            input = X[s]
            result,layer_1_result,layer_2_result = forward(input)
            ans = one_hot_encoding(y[s])
            if result.index(max(result)) == y[s]:
                correct_count += 1
            train(input,layer_1_result,layer_2_result,result,ans)
        print(f"============TEST CASE {i} DONE============")
        print(f"correct_rate: {correct_count/len(X)*100:.2f}%")
    save_weights()


def run_2():
    load_weights()
    X, y = load_digits_data("digits_dataset_drawn.csv")
    for s in range(len(X)):
        input = X[s]
        result,layer_1_result,layer_2_result = forward(input)
        print(f"prediction_result: {result.index(max(result))}")
        print(f"correct_ans: {y[s]}")

def benchmark_mnist():

    def bar(rate, width=28):
        n = int(max(0.0, min(1.0, rate)) * width)
        return "#" * n + "-" * (width - n)

    def section(title):
        line = "=" * 68
        print(f"\n{line}\n{title}\n{line}")

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
    gen_random_weight()
    gen_random_bias()

    section("MNIST BENCHMARK START")
    print(
        f"train={len(X_train)} / test={len(X_test)} / "
        f"features={X_train.shape[1]} / epochs={TRAIN_EPOCHS} / lr={STUDY_RATE}"
    )


    last_train_acc = 0.0

    for i in range(TRAIN_EPOCHS):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        correct_count = 0
        for s in range(len(X_train)):
            input = X_train[s]
            result,layer_1_result,layer_2_result = forward(input)
            ans = one_hot_encoding(y_train[s])
            if result.index(max(result)) == y_train[s]:
                correct_count += 1
            train(input,layer_1_result,layer_2_result,result,ans)

        last_train_acc = correct_count / len(X_train)
        print(
            f"[Epoch {i + 1:02d}/{TRAIN_EPOCHS:02d}] "
            f"train_acc={last_train_acc * 100:6.2f}% "
            f"[{bar(last_train_acc)}] "
        )


    section("EVALUATION")
    correct_count = 0
    for s in range(len(X_test)):
        input = X_test[s]
        result,layer_1_result,layer_2_result = forward(input)
        if result.index(max(result)) == y_test[s]:
            correct_count += 1

    save_weights()

    section("BENCHMARK RESULT")
    print(f"epochs_done     : {TRAIN_EPOCHS}")
    print(f"final_train_acc : {last_train_acc * 100:6.2f}%")
    print(f"final_test_acc  : {correct_count / len(X_test) * 100:6.2f}%")
benchmark_mnist()