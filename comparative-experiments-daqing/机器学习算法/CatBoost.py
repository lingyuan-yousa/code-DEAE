from catboost import CatBoostClassifier, Pool

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


from data_loader import load_data


x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)


train_pool = Pool(x_label, y_label)
test_pool = Pool(x_test, y_test)

# lr = 0.033, 0.027, 0.046
lr = 0.033

model = CatBoostClassifier(
    learning_rate=lr,
    l2_leaf_reg=2,
    n_estimators=500,
    loss_function='MultiClass',
    verbose=False
)

model.fit(train_pool)

y_pred = model.predict(test_pool)

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
