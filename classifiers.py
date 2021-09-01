# coding utf -8
import numpy as np
from scipy.stats import multivariate_normal


class Classifier:
    def __init__(self):
        pass

    def train(self, train_data, train_labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError


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

    
class SVMClassifier(Classifier):
    def __init__(self, dimension, class_num, C, max_iter, kernel_type="poly", sigma=None, d=1, toler=1e-3):
        super().__init__()
        self.dimension = dimension
        self.class_num = class_num
        self.max_iter = max_iter
        self.C = C
        self.toler = toler
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.d = d
        self.train_data = None
        self.train_labels_of = None
        self.b_of = None
        self.alphas_of = None

    def train(self, train_data, train_labels):
        num_samples = train_labels.size
        self.train_data = train_data
        self.train_labels_of = -np.ones((self.class_num, num_samples))
        for c in range(self.class_num):
            self.train_labels_of[c][train_labels == c] = 1

        if self.kernel_type == "rbf":
            K = np.exp(-((np.expand_dims(train_data, 2) - np.expand_dims(train_data.T, 0)) ** 2).sum(1) /
                       (2 * self.sigma**2))
        else:
            K = np.dot(train_data, train_data.T) ** self.d

        self.alphas_of = np.zeros((self.class_num, num_samples))
        self.b_of = np.zeros(self.class_num)
        iter_id = 0
        while iter_id < self.max_iter:
            changed = False
            for i in range(num_samples):
                E_i_of = np.dot(self.alphas_of * self.train_labels_of, K[i]) + self.b_of - \
                         self.train_labels_of[:, i]
                E_j_of = np.zeros_like(E_i_of)
                for c in range(self.class_num):
                    if (self.train_labels_of[c, i] * E_i_of[c] < - self.toler and self.alphas_of[c, i] < self.C) or \
                       (self.train_labels_of[c, i] * E_i_of[c] > self.toler and self.alphas_of[c, i] > 0):
                        j = (i + np.random.randint(1, num_samples)) % num_samples
                        E_j_of[c] = (np.dot(self.alphas_of[c] * self.train_labels_of[c], K[j]) + self.b_of[c] -
                                     self.train_labels_of[c, j])
                        if self.train_labels_of[c, j] == self.train_labels_of[c, i]:
                            min_alpha = max(0, self.alphas_of[c, i] + self.alphas_of[c, j] - self.C)
                            max_alpha = min(self.C, self.alphas_of[c, i] + self.alphas_of[c, j])
                        else:
                            min_alpha = max(0, self.alphas_of[c, j] - self.alphas_of[c, i])
                            max_alpha = min(self.C, self.C + self.alphas_of[c, j] - self.alphas_of[c, i])
                        if min_alpha == max_alpha:
                            continue
                        eta = (K[i, j] * 2 - K[i, i] - K[j, j])
                        if eta >= 0:
                            continue
                        unclipped_alpha_j = (self.alphas_of[c, j] -
                                             self.train_labels_of[c, j] * (E_i_of[c] - E_j_of[c]) / eta)
                        old_alpha_j = self.alphas_of[c, j]
                        self.alphas_of[c, j] = np.clip(unclipped_alpha_j, min_alpha, max_alpha)
                        if abs(self.alphas_of[c, j] - old_alpha_j) < 1e-5:
                            continue
                        old_alpha_i = self.alphas_of[c, i]
                        self.alphas_of[c, i] += (self.train_labels_of[c, j] *
                                                 self.train_labels_of[c, i] *
                                                 (old_alpha_j - self.alphas_of[c, j]))

                        b_1 = (self.b_of[c] - E_i_of[c] -
                               self.train_labels_of[c, i] * (self.alphas_of[c, i] - old_alpha_i) * K[i, i] -
                               self.train_labels_of[c, j] * (self.alphas_of[c, j] - old_alpha_j) * K[i, j])

                        b_2 = (self.b_of[c] - E_j_of[c] -
                               self.train_labels_of[c, i] * (self.alphas_of[c, i] - old_alpha_i) * K[i, j] -
                               self.train_labels_of[c, j] * (self.alphas_of[c, j] - old_alpha_j) * K[j, j])

                        if 0 < self.alphas_of[c, i] < self.C:
                            self.b_of[c] = b_1
                        elif 0 < self.alphas_of[c, j] < self.C:
                            self.b_of[c] = b_2
                        else:
                            self.b_of[c] = (b_1 + b_2) / 2

                        changed = True

                if changed:
                    iter_id = 0
                else:
                    iter_id += 1

    def predict(self, test_data):
        if self.kernel_type == "rbf":
            K = np.exp(-((np.expand_dims(self.train_data, 2) - np.expand_dims(test_data.T, 0)) ** 2).sum(1) /
                       (2 * self.sigma**2))
        else:
            K = np.dot(self.train_data, test_data.T) ** self.d
        return (np.dot(self.alphas_of * self.train_labels_of, K) + np.expand_dims(self.b_of, 1)).argmax(0)
