# coding utf -8
import numpy as np
from scipy.stats import multivariate_normal


class Classifier:
    def __init__(self):
        pass

    def train(self, train_data, train_labels):
        pass

    def predict(self, data):
        pass


class BayesClassifier(Classifier):
    def __init__(self, dimension, class_num):
        super().__init__()
        self.dimension = dimension
        self.class_num = class_num
        self.norms = None
        self.prior_p = None

    def train(self, train_data, train_labels):
        train_data_of_diff_classes = [train_data[train_labels == i] for i in range(self.class_num)]
        self.norms = []
        self.prior_p = np.zeros(self.class_num)
        for i in range(self.class_num):
            self.norms.append(multivariate_normal(mean=np.mean(train_data_of_diff_classes[i], 0),
                                                  cov=np.var(train_data_of_diff_classes[i], 0)))
            self.prior_p[i] = train_data_of_diff_classes[i].shape[0] / train_labels.shape[0]

    def predict(self, test_data):  # 因为后验概率分母一样，就直接比较分子了
        tmp = np.array([norm.pdf(test_data) for norm in self.norms]) * np.expand_dims(self.prior_p, 1)
        return np.argmax(tmp, 0)


class FisherClassifier(Classifier):
    def __init__(self, dimension, class_num):
        super().__init__()
        self.dimension = dimension
        self.class_num = class_num
        self.ws = np.zeros((self.class_num, self.dimension))
        self.y_ts = np.zeros(self.class_num)

    def train(self, train_data, train_labels):
        train_data_of_diff_classes = []
        for _ in range(self.class_num):
            train_data_of_diff_classes.append([])
        for idx, train_label in enumerate(train_labels):
            train_data_of_diff_classes[int(train_label)].append(train_data[idx])
        train_data_of_diff_classes = [np.array(i) for i in train_data_of_diff_classes]
        for i in range(self.class_num):
            train_data_one_class = train_data_of_diff_classes[i]
            train_data_other_class = train_data_of_diff_classes[:]
            train_data_other_class.pop(i)
            train_data_other_class = np.concatenate(train_data_other_class, 0)

            u0 = np.mean(train_data_one_class, 0).reshape(-1, 1)
            u1 = np.mean(train_data_other_class, 0).reshape(-1, 1)

            tmp = train_data_one_class - u0.reshape(1, -1)
            sigma_0 = np.dot(tmp.T, tmp)

            tmp = train_data_other_class - u1.reshape(1, -1)
            sigma_1 = np.dot(tmp.T, tmp)

            s_w = sigma_0 + sigma_1
            w = np.dot(np.linalg.inv(s_w), u0 - u1)
            self.ws[i] = w.squeeze()
            y_t = (np.dot(w.T, u0) + np.dot(w.T, u1)) / 2
            self.y_ts[i] = y_t

    def predict(self, test_data):
        predict_labels = np.zeros(test_data.shape[0], int)
        for idx, sample in enumerate(test_data):
            predict_labels[idx] = np.argmax([np.dot(sample, self.ws[i]) - self.y_ts[i] for i in range(self.class_num)])
        return predict_labels
