import random
#dataset読み込み用
import numpy as np
import json

GRID_NUM = 64
HIDDEN_LAYER_NUM = 2
HIDDEN_LAYER_NODE_NUM = 16
FIRST_LAYER_DIMENSTION = 64
NORM_LAYER_DIMENSTION = 16
STUDY_RATE = 0.001
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
    if raw_data.shape[1] < 65:
        print("エラー: CSVの列数が不足しています。65列(64特徴量+1ラベル)必要です。")
        return None, None

    X = raw_data[:, :64]

    y = raw_data[:, 64].astype(np.int32)
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
    X, y = load_digits_data("digits_dataset.csv")
    for i in range(100):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        for s in range(len(X)):
            input = X[s]
            result,layer_1_result,layer_2_result = forward(input)
            ans = one_hot_encoding(y[s])
            print(f"prediction_result: {result.index(max(result))}")
            print(f"correct_ans: {y[s]}")
            train(input,layer_1_result,layer_2_result,result,ans)
        print(f"============TEST CASE {i} DONE============")
    save_weights()


def run_2():
    load_weights()
    X, y = load_digits_data("digits_dataset_drawn.csv")
    for s in range(len(X)):
        input = X[s]
        result,layer_1_result,layer_2_result = forward(input)
        print(f"prediction_result: {result.index(max(result))}")
        print(f"correct_ans: {y[s]}")
run_2()