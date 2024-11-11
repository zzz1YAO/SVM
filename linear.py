from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def linear_SVM(dataset_path):

    # 假设您的数据集是CSV文件，已经下载到本地
    # 请替换以下路径为您数据集的实际路径

    # 读取数据集
    data = pd.read_csv(dataset_path)

    # 假设最后一列是标签，其余的是特征
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建SVM模型，使用线性核
    model = svm.SVC(kernel='linear')

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    predictions = model.predict(X_test)

    # 打印性能指标
    print(classification_report(y_test, predictions))
    print(f'模型准确度: {accuracy_score(y_test, predictions):.2f}')
def linear_SVM1(dataset_path):

    # 假设您的数据集是CSV文件，已经下载到本地
    # 请替换以下路径为您数据集的实际路径

    # 读取数据集
    data = pd.read_csv(dataset_path)

    # 假设最后一列是标签，其余的是特征
    # 假设第一列是标签，其余的是特征
    X = data.iloc[:, 1:].values  # 选择从第二列到最后一列作为特征
    y = data.iloc[:, 0].values  # 选择第一列作为标签

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建SVM模型，使用线性核
    model = svm.SVC(kernel='linear')

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    predictions = model.predict(X_test)

    # 打印性能指标
    print(classification_report(y_test, predictions))
    print(f'模型准确度: {accuracy_score(y_test, predictions):.2f}')