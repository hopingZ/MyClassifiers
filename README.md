# MyClassifiers
这是一些分类器的简单实现，包括：
- 贝叶斯分类器
- 线性判别分析(Fisher)分类器


# Examples
```
from classifiers import FisherClassifier, BayesClassifier

classifier = FisherClassifier(
  dimension=d,  # 特征有几维
  class_num=C   # 一共有几类
)

classifier.train(
  train_data=train_data,     # nparray (N, d)
  train_labels=train_labels  # nparray (N)
)

predict_labels = classifier.predict(
  test_data=test_data  # nparray (N, d)
)  # -> nparray (N)，输出每个样本所属的类别
```
