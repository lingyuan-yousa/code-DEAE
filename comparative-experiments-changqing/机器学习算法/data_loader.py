import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

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