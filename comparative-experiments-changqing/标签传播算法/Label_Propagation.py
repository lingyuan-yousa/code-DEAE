import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def load_data(label_data_rate):
    seed = 15
    np.random.seed(seed)


    df = pd.read_csv('changqing.csv', encoding='utf-8')  # 或使用你知道的正确编码
    max_min = preprocessing.StandardScaler()

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # 选择特定列作为特征
    x_train = df_train.iloc[:, 2:13].values
    x_train = max_min.fit_transform(x_train)
    y_train = df_train['LITH'].values

    x_test = df_test.iloc[:, 2:13].values
    x_test = max_min.transform(x_test)  # 应使用相同的scaler对象来转换测试数据
    y_test = df_test['LITH'].values

    y_train -= 1  # 标签减一是为了从0开始编码
    y_test -= 1

    idx = np.random.permutation(x_train.shape[0])

    label_idx = idx[:int(len(idx) * label_data_rate)]
    unlab_idx = idx[int(len(idx) * label_data_rate):]

    x_label = x_train[label_idx, :]
    y_label = y_train[label_idx]

    x_unlab = x_train[unlab_idx, :]

    return x_label, y_label, x_unlab, x_test, y_test

# 假设我们使用一部分数据作为标记数据
x_label, y_label, x_unlab, x_test, y_test = load_data(0.1)

# 初始化标签传播模型
label_prop_model = LabelPropagation(max_iter=1500, kernel='rbf', gamma=15)

# 结合标记和未标记的数据
x_total = np.vstack((x_label, x_unlab))
y_total = np.concatenate((y_label, -np.ones(x_unlab.shape[0])))

# 训练模型
label_prop_model.fit(x_total, y_total)

# 进行预测
y_pred = label_prop_model.predict(x_test)


# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
# 计算精确率
precision = precision_score(y_test, y_pred, average='weighted')
# 计算召回率
recall = recall_score(y_test, y_pred, average='weighted')
# 计算 F1 分数
f1_score_value = f1_score(y_test, y_pred, average='weighted')

print(f'准确率：{accuracy:.4f}')
print(f'精确率：{precision:.4f}')
print(f'召回率：{recall:.4f}')
print(f'F1 分数：{f1_score_value:.4f}')

report = classification_report(y_test, y_pred)
print(report)
