# MyClassifiers
这是一些分类器的简单实现，包括：
- 贝叶斯分类器
- 线性判别分析(Fisher)分类器
- 支持向量机


# Examples
```
from classifiers import FisherClassifier, BayesClassifier, SVMClassifier

classifier = FisherClassifier(
  dimension=d,  # 特征有几维
  class_num=C   # 一共有几类
)

'''
classifier = SVMClassifier(
    dimension=4,        # 特征有几维
    class_num=3,        # 一共有几类
    max_iter=100,       # 迭代次数
    C=1e3,              # 惩罚因子
    kernel_type="rbf",  # 核函数类型，可选 "poly", "rbf"
    # d=2,              # "poly"核的幂次
    sigma=1,            # "rbf"核的sigma
)
'''

classifier.train(
  train_data=train_data,     # nparray (N, d)
  train_labels=train_labels  # nparray (N)
)

predict_labels = classifier.predict(
  test_data=test_data  # nparray (N, d)
)  # -> nparray (N)，输出每个样本所属的类别
```
