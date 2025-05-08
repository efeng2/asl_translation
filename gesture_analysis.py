import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras

# 读取训练集和测试集
train_set = pd.read_csv('train_set.csv')
test_set = pd.read_csv('test_set.csv')

# 把训练集和测试集中的因变量和自变量进行拆分
X_train = train_set.iloc[:, 2:]
y_train = train_set['gesture_class']
X_test = test_set.iloc[:,2:]
y_test = test_set['gesture_class']

# 1.SVM支持向量机-SVC
# 参数的设定，交叉检验的设置和结果展示
parameters1 = {'kernel':[ 'rbf', 'linear'], 'C':[0.5, 1, 100], 'gamma':[0.005, 0.1, 0.5, 0.7]}
svc1 = GridSearchCV(SVC(), parameters1).fit(X_train, y_train)
svc1_result = pd.DataFrame.from_dict(svc1.cv_results_)
svc1_result

# SVC最优参数组合
print(svc1.best_params_)

print("rbf核函数的模型训练集得分为：{:.3f}".format(svc1.score(X_train, y_train)))
print("rbf核函数的模型测试集得分为：{:.3f}".format(svc1.score(X_test, y_test)))

# 预测结果表格式展现
svc_predict = svc1.predict(X_test)
dic_SVC = {'Predicted': svc_predict,
            'Experimental': y_test}
df_SVC = pd.DataFrame(dic_SVC)
print(df_SVC)
print(classification_report(y_test, svc_predict))

# 2. XGBoost

# 参数的设定，交叉检验的设置和结果展示
parameters2 = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05], 'max_depth': [3, 5]}
xgb1 = GridSearchCV(XGBClassifier(), parameters2, cv=3).fit(X_train, y_train)
print(xgb1.best_params_)

xgb_predict = xgb1.predict(X_test)
dic_XGB = {'Predicted': xgb_predict,
            'Experimental': y_test}
df_XGB = pd.DataFrame(dic_XGB)
print(df_XGB)
print(classification_report(y_test, xgb_predict))

# 3.神经网络
print(X_train.shape)

# 构建神经网络模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(21, )),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1024, activation='relu'),
    keras.layers.Dense(25, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练神经网络
NN = model.fit(X_train, y_train, epochs=100)

# 评估准确性
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)

# 预测结果表格式展现
nn_predict = model.predict(X_test)
NN_predict = []
for n in range(143):
    predict = np.argmax(nn_predict[n])
    NN_predict.append(predict)
print(NN_predict)

dic_NN = {'Predicted': NN_predict,
            'Experimental': y_test}
df_NN = pd.DataFrame(dic_NN)
print(df_NN)
print(classification_report(y_test, NN_predict))