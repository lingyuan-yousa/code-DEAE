import lightgbm as lgb
import numpy as np

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


from data_loader import load_data


x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)


# 设置 LightGBM 的参数
params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_label)),  # 你的类别数量
    'learning_rate':0.01,
    'num_leaves': 35,  # 适当的叶子节点数量
    'max_depth': -1,  # 适当的最大深度
    'min_child_samples': 20,  # 适当的最小子样本数
    'subsample': 0.8,  # 适当的子采样比例
    'colsample_bytree': 0.8,  # 适当的列采样比例
    'reg_alpha': 1,  # 适当的正则化参数
    'reg_lambda': 1,  # 适当的正则化参数
}

# 初始化 LightGBM 模型
model = lgb.LGBMClassifier(**params, n_estimators=100)

# 训练 LightGBM 模型
model.fit(x_label, y_label)

# 在测试集上进行预测
y_pred = model.predict(x_test)

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
