from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


from data_loader import load_data


x_label, y_label, x_unlab, x_test, y_test = load_data(0.3)


svm_model = SVC(kernel='rbf', C=0.5)

# 训练SVM模型
svm_model.fit(x_label, y_label)

# 在测试集上进行预测
y_pred = svm_model.predict(x_test)

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
