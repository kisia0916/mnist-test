import random
GRID_NUM = 64
HIDDEN_LAYER_NUM = 2
HIDDEN_LAYER_NODE_NUM = 16
FIRST_LAYER_DIMENSTION = 64
NORM_LAYER_DIMENSTION = 16
STUDY_RATE = 0.1
FIRST_INPUT = [random.randint(0,1) for _ in range(GRID_NUM)]
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


def forward():
    now_vec = FIRST_INPUT.copy()
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
        result.append(calc_dot(now_vec,ans_weight[s])-biases[-1][s])
    return result,layer_1_result,layer_2_result

def train(layer_1_result,layer_2_result,out_put,ans):
    diffs = [out_put[i]-ans[i] for i in range(len(ans))]

    layer_2_diff = []
    for i in range(NORM_LAYER_DIMENSTION):
        transposed = []
        for s in range(10):
            transposed.append(ans_weight[s][i])
        if layer_2_result[i] != 0:
            layer_2_diff.append(calc_dot(transposed,diffs))
        else:
            layer_2_diff.append(0)

    layer_1_diff = []
    for i in range(HIDDEN_LAYER_NODE_NUM):
        if layer_1_result[i] != 0:
            layer_1_diff.append(calc_dot(hidden_layer[1][i],layer_2_diff))
        else:
            layer_1_diff.append(0)


    print(diffs)
    print(layer_2_result)
    print(layer_1_result)
    for i in range(10):
        error = out_put[i] - ans[i]
        biases[-1][i] = biases[-1][i] - (STUDY_RATE*error)
        for s in range(NORM_LAYER_DIMENSTION):
            ans_weight[i][s] = ans_weight[i][s] - (STUDY_RATE*error*layer_2_result[s])

def main():
    gen_random_weight()
    gen_random_bias()
    result,layer_1_result,layer_2_result = forward()
    #print(result)
    ans = [random.randint(0,1) for _ in range(10)]
    train(layer_1_result,layer_2_result,result,ans)
main()