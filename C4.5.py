import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
# 假设您已经加载了DataFrame到变量data中
# data = pd.read_csv('your_data.csv')  # 如果是CSV文件
# data = pd.read_excel('your_data.xlsx')  # 如果是Excel文件
# 或者其他数据加载方法
data=pd.read_csv("Iris/iris.data")
# 分离特征和目标变量
X = data.iloc[:, :-1]  # 选择除了最后一列的所有列作为特征
y = data.iloc[:, -1]   # 选择最后一列作为标签

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# 创建决策树模型（C4.5相当于sklearn中的“entropy”选项）
clf = DecisionTreeClassifier(criterion='entropy')

# 训练模型
clf.fit(X_train, y_train)

# 绘制决策树
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=y.unique().astype(str), rounded=True)
plt.show()



