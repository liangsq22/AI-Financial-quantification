# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 23:45:07 2024

@author: 梁思奇
"""

###输出警告改动：屏蔽警告 
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 改动：只显示错误信息 

###############################8.1第一个示例

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pprint import pprint
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision = 4
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'


def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
set_seeds()

a = np.arange(100)
print("\n 8.1")
print("原始数据a（0-99）:")
print(a)

a = a.reshape((len(a), -1)) #转换为2维数据，行维表示时间步长，列维表示特征
print("\n重塑后的数据形状:")
print(a.shape)
print("\n重塑后的数据前5行:")
print(a[:5])

#通过TimeseriesGenerator创建一批滞后数据
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

lags = 3
#生成训练样本集，以元组为元素的列表，其中元组包括2个数组为元素，1个是滞后值（特征）
#另一个是预测值（标签）
#因为batch_size=5，所以第一数组形状是（1，3，1），表示3个时间步长，1个特征
#第2个数组形状是（5，1）
#因为每个批次5个样本，总共20个批次，共97个样本，最后批次2个样本
g = TimeseriesGenerator(a, a, length=lags, batch_size=5) #批量是5
print("\nTimeseriesGenerator的总批次数:")
print(len(g))

## 改动开始
# pprint(list(g)[0])
# pprint(list(g)[-1])

# 直接打印第一个和最后一个，不使用list
for i, batch in enumerate(g):
    if i == 0:  # 打印第一个批次
        print("第一批次")
        pprint(batch)
    elif i == len(g) - 1:  # 打印最后一个批次
        print("最后批次")
        pprint(batch)
        break  # 打印最后一个批次后退出循环
## 改动结束


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense

model = Sequential()
#用SimpleRNN作为单一隐藏层
model.add(SimpleRNN(100, activation='relu',
                    #注意：input_shape的形状
                    input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adagrad', loss='mse',
              metrics=['mae'])

#浅层RNN模型描述信息
#simple_rnn的参数数量：神经元数量*神经元数量+（特征数量+偏置数量）*神经元数量
model.summary()

#基于生成器对象对RNN进行拟合
#每轮次使用5个批次样本
h = model.fit(g, epochs=1000, steps_per_epoch=5,
            verbose=False)

res = pd.DataFrame(h.history)
print("\n训练历史的最后4行:")
print(res.tail(4))
res.iloc[10:].plot(figsize=(10, 6), style=['--', '--'])  # 绘图1
plt.title('Training Loss and MAE Trend')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(['Loss (loss)', 'Mean Absolute Error (mae)'])
plt.show()  # 输出图1
res.iloc[10:100].plot(figsize=(10, 6), style=['--', '--'])  # 绘图2
plt.title('Training Loss and MAE Trend（10-100）')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(['Loss (loss)', 'Mean Absolute Error (mae)'])
plt.show()  # 输出图2

# 样本内预测
# 注意x的形状需要上升为3维
x_in = np.array([21, 22, 23]).reshape((1, lags, 1))
y_in = model.predict(x_in, verbose=False)
theoretical_in = 24  # 理论答案
pred_in = int(round(y_in[0, 0]))
print("\n样本内预测:")
print(f"输入数据: {x_in.flatten().astype(int)}")
print(f"理论答案: {theoretical_in}")
print(f"实际预测值: {pred_in}")

# 样本外预测
x_out = np.array([87, 88, 89]).reshape((1, lags, 1))
y_out = model.predict(x_out, verbose=False)
theoretical_out = 90  # 理论答案
pred_out = int(round(y_out[0, 0]))
print("\n样本外预测:")
print(f"输入数据: {x_out.flatten().astype(int)}")
print(f"理论答案: {theoretical_out}")
print(f"实际预测值: {pred_out}")

# 远离样本的预测
x_far1 = np.array([187, 188, 189]).reshape((1, lags, 1))
y_far1 = model.predict(x_far1, verbose=False)
theoretical_far1 = 190  # 理论答案
pred_far1 = int(round(y_far1[0, 0]))
print("\n远离样本的预测1:")
print(f"输入数据: {x_far1.flatten().astype(int)}")
print(f"理论答案: {theoretical_far1}")
print(f"实际预测值: {pred_far1}")

x_far2 = np.array([1187, 1188, 1189]).reshape((1, lags, 1))
y_far2 = model.predict(x_far2, verbose=False)
theoretical_far2 = 1190  # 理论答案
pred_far2 = int(round(y_far2[0, 0]))
print("\n远离样本的预测2:")
print(f"输入数据: {x_far2.flatten().astype(int)}")
print(f"理论答案: {theoretical_far2}")
print(f"实际预测值: {pred_far2}")




#########################8.2第二个示例
def transform(x):
    y = 0.05 * x ** 2 + 0.2 * x + np.sin(x) + 5 #确定性变换
    y += np.random.standard_normal(len(x)) * 0.2  #随机变换
    return y

x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
a = transform(x)

plt.figure(figsize=(10, 6))
plt.plot(x, a)  ### 输出图像
# 添加标题和轴标签
plt.title('Transformed Time Series Data with Deterministic and Stochastic Components', fontsize=14)
plt.xlabel('X-axis (radians)', fontsize=14)
plt.ylabel('Transformed Y-axis', fontsize=14)
# 显示图像
plt.show()


###TimeseriesGenerator对原始数据进行变换

a = a.reshape((len(a), -1))
print("\n 8.2")
print("重塑数据a的前5行")
print(a[:5])
lags = 5
g = TimeseriesGenerator(a, a, length=lags, batch_size=5)

model = Sequential()
model.add(SimpleRNN(500, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

# 显示模型信息
model.summary()

model.fit(g, epochs=500,
          steps_per_epoch=10,
          verbose=False)


###构建预测区间

x = np.linspace(-6 * np.pi, 6 * np.pi, 1000)
d = transform(x)

#批次大小为len(d)，所以只有1个批次
g_ = TimeseriesGenerator(d, d, length=lags, batch_size=len(d))

## 改动开始
# f = list(g_)[0][0].reshape((len(d) - lags, lags, 1))

# 获取生成器中的第一个批次
for x_batch, y_batch in g_:
    # 将 x_batch 进行 reshape，使其形状变为 (len(d) - lags, lags, 1)
    f = x_batch.reshape((len(d) - lags, lags, 1))
    break  # 只获取第一个批次，所以这里使用 break 跳出循环
## 改动结束

y = model.predict(f, verbose=False) #样本内和样本外预测

### 输出图像
plt.figure(figsize=(10, 6))
plt.title('Model Predictions vs Data (In-sample and Out-of-sample)', fontsize=14)
plt.plot(x[lags:], d[lags:], label='data', alpha=0.75)
plt.plot(x[lags:], y, 'r.', label='pred', ms=3)
plt.axvline(-2 * np.pi, c='g', ls='--')
plt.axvline(2 * np.pi, c='g', ls='--')
plt.text(-15, 22, 'out-of-sample')
plt.text(-2, 22, 'in-sample')
plt.text(10, 22, 'out-of-sample')
plt.legend()




#########################8.3金融价格序列：用SimpleRNN和LSTM

# 加载新数据集 TRD_Dalyr_Moutai.csv
raw = pd.read_csv('TRD_Dalyr_Moutai.csv', index_col='Trddt', parse_dates=True)
symbol = 'Moutai_Close_Price'  # 重命名为更具描述性的名称

### 导入数据并重新采样
def generate_data():
    data = pd.DataFrame(raw['Clsprc'])  # 选择 'Clsprc' 列
    data.columns = [symbol]  # 将该列重命名
    # 由于新数据是日级别的，不需要重新采样
    # 如果有缺失值，可以进行填充
    data = data.ffill()
    return data

data = generate_data()
data = (data - data.mean()) / data.std()#高斯归一化
p = data[symbol].values
p = p.reshape((len(p), -1))#数据重塑为2维

###对RNN模型训练
lags = 5
g = TimeseriesGenerator(p, p, length=lags, batch_size=5)

def create_rnn_model(hu=100, lags=lags, layer='SimpleRNN',
                           features=1, algorithm='estimation'):
    model = Sequential()
    #增加一个SimpleRNN层或LSTM层
    if layer == 'SimpleRNN': 
        model.add(SimpleRNN(hu, activation='relu',
                            input_shape=(lags, features)))
    else:
        model.add(LSTM(hu, activation='relu',
                       input_shape=(lags, features)))
    #增加一个用于估计或分类的输出层
    if algorithm == 'estimation': 
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
    return model


# 使用 SimpleRNN 模型
model_rnn = create_rnn_model(layer='SimpleRNN')
model_rnn.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# 使用 LSTM 模型
model_lstm = create_rnn_model(layer='LSTM')
model_lstm.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# 样本内预测 - SimpleRNN
y_rnn = model_rnn.predict(g, verbose=False).flatten()  # SimpleRNN 预测
data['pred_rnn'] = np.nan
data['pred_rnn'].iloc[lags:] = y_rnn

print("\n 8.3")
print(f"生成器中的批次数量: {len(g)}")
print(f"预测值的数量: {len(y_rnn)}")  # 生成g是batch_size=5, 每个批次5个样本，len(y)=5*len(g)

# 样本内预测 - LSTM
y_lstm = model_lstm.predict(g, verbose=False).flatten()  # LSTM 预测
data['pred_lstm'] = np.nan
data['pred_lstm'].iloc[lags:] = y_lstm

from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# 计算方向准确度 - SimpleRNN
direction_accuracy_rnn = accuracy_score(np.sign(data[symbol].iloc[lags:]), np.sign(data['pred_rnn'].iloc[lags:]))
print(f"\nSimpleRNN 样本内预测方向准确度: {direction_accuracy_rnn:.4f}")

# 计算统计指标 - SimpleRNN
mae_rnn = mean_absolute_error(data[symbol].iloc[lags:], data['pred_rnn'].iloc[lags:])
mse_rnn = mean_squared_error(data[symbol].iloc[lags:], data['pred_rnn'].iloc[lags:])
rmse_rnn = np.sqrt(mse_rnn)

# 输出统计指标 - SimpleRNN
print(f"\nSimpleRNN 模型均绝对误差 (MAE): {mae_rnn:.4f}")
print(f"SimpleRNN 模型均方误差 (MSE): {mse_rnn:.4f}")
print(f"SimpleRNN 模型均方根误差 (RMSE): {rmse_rnn:.4f}")

# 计算方向准确度 - LSTM
direction_accuracy_lstm = accuracy_score(np.sign(data[symbol].iloc[lags:]), np.sign(data['pred_lstm'].iloc[lags:]))
print(f"\nLSTM 样本内预测方向准确度: {direction_accuracy_lstm:.4f}")

# 计算统计指标 - LSTM
mae_lstm = mean_absolute_error(data[symbol].iloc[lags:], data['pred_lstm'].iloc[lags:])
mse_lstm = mean_squared_error(data[symbol].iloc[lags:], data['pred_lstm'].iloc[lags:])
rmse_lstm = np.sqrt(mse_lstm)

# 输出统计指标 - LSTM
print(f"\nLSTM 模型均绝对误差 (MAE): {mae_lstm:.4f}")
print(f"LSTM 模型均方误差 (MSE): {mse_lstm:.4f}")
print(f"LSTM 模型均方根误差 (RMSE): {rmse_lstm:.4f}")

# 全部数据图像 - SimpleRNN
data[[symbol, 'pred_rnn']].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75)
plt.title('Financial Price Series Prediction (SimpleRNN)')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签：时间步长
plt.ylabel(f'{symbol} and Predicted')  # Y轴标签：股票价格与预测值
plt.show()

# 放大视图 - SimpleRNN (100到200时间步)
data[[symbol, 'pred_rnn']].iloc[100:200].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75)
plt.title('Zoomed View: Time Steps 100 to 200 (SimpleRNN)')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签：时间步长
plt.ylabel(f'{symbol} and Predicted')  # Y轴标签：股票价格与预测值
plt.show()

# 全部数据图像 - LSTM
data[[symbol, 'pred_lstm']].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75)
plt.title('Financial Price Series Prediction (LSTM)')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签：时间步长
plt.ylabel(f'{symbol} and Predicted')  # Y轴标签：股票价格与预测值
plt.show()

# 放大视图 - LSTM (100到200时间步)
data[[symbol, 'pred_lstm']].iloc[100:200].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75)
plt.title('Zoomed View: Time Steps 100 to 200 (LSTM)')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签：时间步长
plt.ylabel(f'{symbol} and Predicted')  # Y轴标签：股票价格与预测值
plt.show()




##########################8.4金融收益率序列：用SimpleRNN
data = generate_data()
data['r'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data = (data - data.mean()) / data.std()
r = data['r'].values
r = r.reshape((len(r), -1))

g = TimeseriesGenerator(r, r, length=lags, batch_size=5)

# 使用 SimpleRNN 模型
model_rnn = create_rnn_model(layer='SimpleRNN')
model_rnn.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# 使用 LSTM 模型
model_lstm = create_rnn_model(layer='LSTM')
model_lstm.fit(g, epochs=500, steps_per_epoch=10, verbose=False)

# 样本内预测 - SimpleRNN
y_rnn = model_rnn.predict(g, verbose=False).flatten()  # SimpleRNN 预测
data['pred_rnn'] = np.nan
data['pred_rnn'].iloc[lags:] = y_rnn

print("\n 8.4")
print(f"生成器中的批次数量: {len(g)}")
print(f"预测值的数量: {len(y_rnn)}")  # 生成g是batch_size=5, 每个批次5个样本，len(y)=5*len(g)

# 样本内预测 - LSTM
y_lstm = model_lstm.predict(g, verbose=False).flatten()  # LSTM 预测
data['pred_lstm'] = np.nan
data['pred_lstm'].iloc[lags:] = y_lstm

# simplernn
# 输出图表曲线
data[['r', 'pred_rnn']].iloc[100:150].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75);
plt.axhline(0, c='grey', ls='--')
plt.title('Financial Return Series Prediction(SimpleRNN) - In-Sample')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签

data.dropna(inplace=True)
direction_accuracy = accuracy_score(np.sign(data['r']), np.sign(data['pred_rnn']))  # 计算方向准确度
print(f"\nSimpleRNN样本内方向预测准确度: {direction_accuracy:.4f}")  # 输出方向准确度

# 样本内评估指标
mae = np.mean(np.abs(data['r'] - data['pred_rnn']))  # 均绝对误差 (MAE)
mse = np.mean((data['r'] - data['pred_rnn']) ** 2)  # 均方误差 (MSE)
rmse = np.sqrt(mse)  # 均方根误差 (RMSE)

print(f"\n均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")

# lstm
# 输出图表曲线
data[['r', 'pred_lstm']].iloc[100:150].plot(
            figsize=(10, 6), style=['b', 'r-.'],
            alpha=0.75);
plt.axhline(0, c='grey', ls='--')
plt.title('Financial Return Series Prediction(LSTM) - In-Sample')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签

direction_accuracy = accuracy_score(np.sign(data['r']), np.sign(data['pred_lstm']))  # 计算方向准确度
print(f"\nLSTM样本内方向预测准确度: {direction_accuracy:.4f}")  # 输出方向准确度

# 样本内评估指标
mae = np.mean(np.abs(data['r'] - data['pred_lstm']))  # 均绝对误差 (MAE)
mse = np.mean((data['r'] - data['pred_lstm']) ** 2)  # 均方误差 (MSE)
rmse = np.sqrt(mse)  # 均方根误差 (RMSE)

print(f"\n均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")


###样本外
split = int(len(r) * 0.8)  # 分训练测试集，80%、20%
train = r[:split]
test = r[split:]

# 使用 SimpleRNN 模型
g_train_rnn = TimeseriesGenerator(train, train, length=lags, batch_size=5)
set_seeds()
model_rnn_out = create_rnn_model(hu=100, layer='SimpleRNN')
model_rnn_out.fit(g_train_rnn, epochs=100, steps_per_epoch=10, verbose=False)  # 训练集拟合模型

g_test_rnn = TimeseriesGenerator(test, test, length=lags, batch_size=5)
y_test_rnn = model_rnn_out.predict(g_test_rnn, verbose=False).flatten()

# 样本外预测指标 - SimpleRNN
direction_accuracy_test_rnn = accuracy_score(np.sign(test[lags:]), np.sign(y_test_rnn))  # 计算方向准确度
print(f"\nSimpleRNN样本外方向预测准确度: {direction_accuracy_test_rnn:.4f}")

mae_test_rnn = np.mean(np.abs(test[lags:] - y_test_rnn))  # 样本外 MAE
mse_test_rnn = np.mean((test[lags:] - y_test_rnn) ** 2)  # 样本外 MSE
rmse_test_rnn = np.sqrt(mse_test_rnn)  # 样本外 RMSE

print(f"\nSimpleRNN 样本外均绝对误差 (MAE): {mae_test_rnn:.4f}")
print(f"SimpleRNN 样本外均方误差 (MSE): {mse_test_rnn:.4f}")
print(f"SimpleRNN 样本外均方根误差 (RMSE): {rmse_test_rnn:.4f}")

# 使用 LSTM 模型
g_train_lstm = TimeseriesGenerator(train, train, length=lags, batch_size=5)
set_seeds()
model_lstm_out = create_rnn_model(hu=100, layer='LSTM')
model_lstm_out.fit(g_train_lstm, epochs=100, steps_per_epoch=10, verbose=False)  # 训练集拟合模型

g_test_lstm = TimeseriesGenerator(test, test, length=lags, batch_size=5)
y_test_lstm = model_lstm_out.predict(g_test_lstm, verbose=False).flatten()

# 样本外预测指标 - LSTM
direction_accuracy_test_lstm = accuracy_score(np.sign(test[lags:]), np.sign(y_test_lstm))  # 计算方向准确度
print(f"\nLSTM样本外方向预测准确度: {direction_accuracy_test_lstm:.4f}")

mae_test_lstm = np.mean(np.abs(test[lags:] - y_test_lstm))  # 样本外 MAE
mse_test_lstm = np.mean((test[lags:] - y_test_lstm) ** 2)  # 样本外 MSE
rmse_test_lstm = np.sqrt(mse_test_lstm)  # 样本外 RMSE

print(f"\nLSTM 样本外均绝对误差 (MAE): {mae_test_lstm:.4f}")
print(f"LSTM 样本外均方误差 (MSE): {mse_test_lstm:.4f}")
print(f"LSTM 样本外均方根误差 (RMSE): {rmse_test_lstm:.4f}")




############################8.5金融特征
data = generate_data()
data['r'] = np.log(data / data.shift(1))
window = 20
data['mom'] = data['r'].rolling(window).mean()#增加时间序列动量特征
data['vol'] = data['r'].rolling(window).std()#增加滚动波动率特征
data.dropna(inplace=True)


#####8.5.1估计
split = int(len(data) * 0.8)
train = data.iloc[:split].copy()
mu, std = train.mean(), train.std()
train = (train - mu) / std
test = data.iloc[split:].copy()
test = (test - mu) / std

lags = 3 # 重新设置

# 创建时间序列生成器，用于预测价格
g_price = TimeseriesGenerator(train[['r', 'mom', 'vol']].values, train['Moutai_Close_Price'].values, length=lags, batch_size=5)

# 创建时间序列生成器，用于预测收益率
g_returns = TimeseriesGenerator(train[['r', 'mom', 'vol']].values, train['r'].values, length=lags, batch_size=5)

print("\n 8.5")
print("训练集收益率样本前10个值：", train['r'].values[:10])
print("训练集价格样本前10个值：", train['Moutai_Close_Price'].values[:10])

set_seeds()
# 创建并训练价格预测模型：手动修改使用SimpleRNN或LSTM
model_price = create_rnn_model(hu=100, lags=lags, features=len(['r', 'mom', 'vol']), layer='SimpleRNN')
model_price.fit(g_price, epochs=100, steps_per_epoch=10, verbose=False)

# 创建并训练收益率预测模型：手动修改使用SimpleRNN或LSTM
model_returns = create_rnn_model(hu=100, lags=lags, features=len(['r', 'mom', 'vol']), layer='LSTM')
model_returns.fit(g_returns, epochs=100, steps_per_epoch=10, verbose=False)

# 模型预测
y_price = model_price.predict(TimeseriesGenerator(test[['r', 'mom', 'vol']].values, test['Moutai_Close_Price'].values, length=lags, batch_size=5)).flatten()
y_returns = model_returns.predict(TimeseriesGenerator(test[['r', 'mom', 'vol']].values, test['r'].values, length=lags, batch_size=5)).flatten()

# 计算价格的方向准确度
price_direction_accuracy = accuracy_score(np.sign(test['Moutai_Close_Price'].iloc[lags:]), np.sign(y_price))
print(f"\n价格的方向准确度: {price_direction_accuracy:.4f}")

# 计算价格的 MSE 和 MAE
price_mse = mean_squared_error(test['Moutai_Close_Price'].iloc[lags:], y_price)
price_mae = mean_absolute_error(test['Moutai_Close_Price'].iloc[lags:], y_price)
print(f"\n价格预测的均方误差 (MSE): {price_mse:.4f}")
print(f"价格预测的平均绝对误差 (MAE): {price_mae:.4f}")

# 计算收益率的方向准确度
return_direction_accuracy = accuracy_score(np.sign(test['r'].iloc[lags:]), np.sign(y_returns))
print(f"\n收益率的方向准确度: {return_direction_accuracy:.4f}")

# 计算收益率的 MSE 和 MAE
return_mse = mean_squared_error(test['r'].iloc[lags:], y_returns)
return_mae = mean_absolute_error(test['r'].iloc[lags:], y_returns)
print(f"\n收益率预测的均方误差 (MSE): {return_mse:.4f}")
print(f"收益率预测的平均绝对误差 (MAE): {return_mae:.4f}")

# 创建一个新的DataFrame来存储测试数据和预测值
test_data_price = test.iloc[lags:].copy()
test_data_price['pred_price'] = y_price  # 预测价格

test_data_returns = test.iloc[lags:].copy()
test_data_returns['pred_returns'] = y_returns  # 预测收益率

# 输出价格预测图表
plt.figure(figsize=(12, 6))
plt.plot(test.index[lags:], test['Moutai_Close_Price'].iloc[lags:], label='Actual Price', color='blue')  # 真实值
plt.plot(test.index[lags:], y_price, label='Predicted Price', color='red')  # 预测值
plt.title('Comparison of Predicted and Actual Prices')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Prices')  # Y轴标签
plt.legend()
plt.show()

# 输出收益率预测图表
plt.figure(figsize=(12, 6))
plt.plot(test.index[lags:], test['r'].iloc[lags:], label='Actual Returns', color='blue')  # 真实值
plt.plot(test.index[lags:], y_returns, label='Predicted Returns', color='red')  # 预测值
plt.title('Comparison of Predicted and Actual Returns')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签
plt.legend()
plt.show()

# 收益率分布图
plt.figure(figsize=(12, 6))
plt.hist(test['r'].iloc[lags:], bins=50, alpha=0.5, label='Actual Return', color='blue')  # 真实值收益率
plt.hist(y_returns, bins=50, alpha=0.5, label='Predicted Return', color='red')  # 预测值收益率
plt.title('Distribution of Actual and Predicted Returns')  # 图表标题
plt.legend()
plt.show()



####8.5.2分类：用SimpleRNN和LSTM测收益率
train_y = np.where(train['r'] > 0, 1, 0)
test_y = np.where(test['r'] > 0, 1, 0)

print("训练集收益率样本前10个值：", train_y[:10])
print("类别分布：")
print(np.bincount(train_y))

def cw(a):
    c0, c1 = np.bincount(a)
    w0 = (1 / c0) * (len(a)) / 2
    w1 = (1 / c1) * (len(a)) / 2
    return {0: w0, 1: w1}

## 改动开始
class CustomTimeseriesGenerator(TimeseriesGenerator):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def __getitem__(self, index):
        X, y = super().__getitem__(index)
        if self.class_weights is not None:
            sample_weights = np.array([self.class_weights[label] for label in y])
        else:
            sample_weights = np.ones_like(y)
        return X, y, sample_weights

# 定义类别权重
class_weights = cw(train_y)
## 改动结束

# 创建自定义生成器
g = CustomTimeseriesGenerator(train.values, train_y, 
                              length=lags, batch_size=5, class_weights=class_weights)
g_ = TimeseriesGenerator(test.values, test_y,
                         length=lags, batch_size=5)

# 训练模型时不再需要传递 class_weight
# simplernn
set_seeds()
model_rnn = create_rnn_model(hu=50,
            features=len(data.columns),
            layer='SimpleRNN',
            algorithm='classification')
model_rnn.fit(g, epochs=5, steps_per_epoch=10, verbose=False)

y_rnn = np.where(model_rnn.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

# lstm
set_seeds()
model_lstm = create_rnn_model(hu=50,
            features=len(data.columns),
            layer='LSTM',
            algorithm='classification')
model_lstm.fit(g, epochs=5, steps_per_epoch=10, verbose=False)

y_lstm = np.where(model_lstm.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

print("SimpleRNN预测后类别分布：")
print(np.bincount(y_rnn))
print("LSTM预测后类别分布：")
print(np.bincount(y_lstm))

direction_accuracy1=accuracy_score(test_y[lags:], y_rnn)
print(f"\nSimpleRNN,分类后收益率预测的方向准确度: {direction_accuracy1:.4f}")

direction_accuracy2=accuracy_score(test_y[lags:], y_lstm)
print(f"\nLSTM,分类后收益率预测的方向准确度: {direction_accuracy2:.4f}")

###市场方向预测表现更好



###8.5.3深度RNN
from tensorflow.keras.layers import Dropout

def create_deep_rnn_model(hl=2, hu=100, layer='SimpleRNN',
                          optimizer='rmsprop', features=1,
                          dropout=False, rate=0.3, seed=100):
    #保证最少2个隐藏层
    if hl <= 2: hl = 2
    if layer == 'SimpleRNN':
        layer = SimpleRNN
    else:
        layer = LSTM
    model = Sequential()
    #第一个隐藏层
    model.add(layer(hu, input_shape=(lags, features),
                     return_sequences=True,
                    ))
    if dropout:
        model.add(Dropout(rate, seed=seed))
    for _ in range(2, hl):
        model.add(layer(hu, return_sequences=True))
        if dropout:
            model.add(Dropout(rate, seed=seed))#Dropout层
    model.add(layer(hu)) #最终隐藏层
    model.add(Dense(1, activation='sigmoid')) #建立分类模型
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


set_seeds()
model1 = create_deep_rnn_model(
            hl=2, hu=50, layer='SimpleRNN',
            features=len(data.columns),
            dropout=True, rate=0.3)

model1.summary()

model1.fit(g, epochs=200, steps_per_epoch=10,
          verbose=False)

y1 = np.where(model1.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

print("类别分布：")
print(np.bincount(y1))

direction_accuracy=accuracy_score(test_y[lags:], y1)
print(f"\nSimpleRNN深度RNN预测的方向准确度: {direction_accuracy:.4f}")

set_seeds()
model2 = create_deep_rnn_model(
            hl=2, hu=50, layer='LSTM',
            features=len(data.columns),
            dropout=True, rate=0.3)

model2.summary()

model2.fit(g, epochs=200, steps_per_epoch=10,
          verbose=False)

y2 = np.where(model2.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

direction_accuracy=accuracy_score(test_y[lags:], y2)
print(f"\nLSTM深度RNN预测的方向准确度: {direction_accuracy:.4f}")

