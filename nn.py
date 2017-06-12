import math
import random
import numpy as np
import yaml

random.seed(0)


def rand(a, b):
    return (b - a) * random.random() + a


def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class NeuralNet:
    def __init__(self):

        self.input_n = 0
        self.hidden_n = 0
        self.output_n = 0

        self.input_cells = []
        self.hidden_cells = []
        self.output_cells = []

        self.input_weights = []
        self.output_weights = []

        self.input_correction = []
        self.output_correction = []

    def setup(self, ni, nh, no):
        self.input_n = ni + 1
        self.hidden_n = nh
        self.output_n = no

        # init cells
        self.input_cells = [1.0] * self.input_n
        self.hidden_cells = [1.0] * self.hidden_n
        self.output_cells = [1.0] * self.output_n

        # init weights
        self.input_weights = make_matrix(self.input_n, self.hidden_n)
        self.output_weights = make_matrix(self.hidden_n, self.output_n)

        # random activate
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                self.input_weights[i][h] = rand(-0.2, 0.2)

        for h in range(self.hidden_n):
            for o in range(self.output_n):
                self.output_weights[h][o] = rand(-2.0, 2.0)

        # init correction matrix
        self.input_correction = make_matrix(self.input_n, self.hidden_n)
        self.output_correction = make_matrix(self.hidden_n, self.output_n)

    def predict(self, inputs):
        # activate input layer
        for i in range(self.input_n - 1):
            self.input_cells[i] = inputs[i]
        # activate hidden layer
        for j in range(self.hidden_n):
            total = 0.0
            for i in range(self.input_n):
                total += self.input_cells[i] * self.input_weights[i][j]
            self.hidden_cells[j] = sigmoid(total)

        # activate output layer
        for k in range(self.output_n):
            total = 0.0
            for j in range(self.hidden_n):
                total += self.hidden_cells[j] * self.output_weights[j][k]
            self.output_cells[k] = sigmoid(total)

        return self.output_cells[:]

    def back_propagate(self, case, label, learn, correct):
        # feed forward
        self.predict(case)

        # get output layer error
        output_deltas = [0.0] * self.output_n

        for o in range(self.output_n):
            error = label[o] - self.output_cells[o]
            output_deltas[o] = sigmoid_derivative(self.output_cells[o]) * error

        # get hidden layer error
        hidden_deltas = [0.0] * self.hidden_n
        for h in range(self.hidden_n):
            error = 0.0
            for o in range(self.output_n):
                error += output_deltas[o] * self.output_weights[h][o]
            hidden_deltas[h] = sigmoid_derivative(self.hidden_cells[h]) * error

        # update output weights
        for h in range(self.hidden_n):
            for o in range(self.output_n):
                change = output_deltas[o] * self.hidden_cells[h]
                self.output_weights[h][o] += learn * change + correct * self.output_correction[h][o]
                self.output_correction[h][o] = change

        # update input weights
        for i in range(self.input_n):
            for h in range(self.hidden_n):
                change = hidden_deltas[h] * self.input_cells[i]
                self.input_weights[i][h] += learn * change + correct * self.input_correction[i][h]
                self.input_correction[i][h] = change

        # get global error
        error = 0.0
        for o in range(len(label)):
            error += 0.5 * (label[o] - self.output_cells[o]) ** 2
        return error

    def read_data(self, filename, has_label):
        mat = np.genfromtxt(filename, delimiter=',')
        mat = mat[1:]
        return mat


    def save(self, filename):
        with open(filename, 'w') as f:
            model_Para = {"input_n": self.input_n,
                          "hidden_n": self.hidden_n,
                          "output_n": self.output_n,
                          "input_weights": self.input_weights,
                          "output_weights": self.output_weights}
            yaml.dump(model_Para, f, default_flow_style=False)

    def load(self, filename):
        model_Para = []
        with open(filename, 'r') as f:
            model_Para = yaml.load(f)

            self.input_n = model_Para["input_n"]
            self.hidden_n = model_Para["hidden_n"]
            self.output_n = model_Para["output_n"]
            self.input_weights = model_Para["input_weights"]
            self.output_weights = model_Para["output_weights"]

    def train(self, data_file="training_ANN.csv", save_file="save.yaml",
              limit=10, learn=0.05, correct=0.1):

        data = self.read_data(data_file, True)
        n = data.shape[0]

        cases = data[:,:2].tolist()
        labels = data[:,2].tolist()

        print  cases

        for i in range(n):
            labels[i] = [data[i][2]]

        print  labels

        print cases

        for j in range(limit):
            error = 0.0
            for i in range(len(cases)):
                label = labels[i]
                case = cases[i]
                error += self.back_propagate(case, label, learn, correct)
            print(error)
        print  self.input_weights
        self.save(save_file)

    def test(self):

        self.setup(2, 5, 1)
        data_file = "training_ANN.csv"
        save_file = "save.yaml"
        self.train(data_file, save_file, 100, 0.05, 0.1)


if __name__ == '__main__':
    nn = NeuralNet()
    nn.test()

    nn.load('save.yaml')
    print  'output weights'
    print  nn.output_weights
    print  'input weights'
    print  nn.input_weights
    print  'nums'
    print  nn.input_n, nn.hidden_n, nn.output_n
