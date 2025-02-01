# -*- coding: utf-8 -*-


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
print("8.1")
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
        pprint(batch)
    elif i == len(g) - 1:  # 打印最后一个批次
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
print("\n训练历史的最后3行:")
print(res.tail(3))
res.iloc[10:].plot(figsize=(10, 6), style=['--', '--'])
plt.title('训练损失和MAE的变化趋势')
plt.xlabel('训练轮次')
plt.ylabel('值')
plt.legend(['损失 (loss)', '均绝对误差 (mae)'])
plt.show()  # 输出图1

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

# 绘制图像
plt.figure(figsize=(12, 6))
plt.plot(x, a)
# 添加标题和轴标签
plt.title('Transformed Time Series Data with Deterministic and Stochastic Components', fontsize=16)
plt.xlabel('X-axis (radians)', fontsize=14)
plt.ylabel('Transformed Y-axis', fontsize=14)
# 显示图像
plt.show()

###TimeseriesGenerator对原始数据进行变换

a = a.reshape((len(a), -1))
print("8.2")
print("重塑数据a的前5行")
print(a[:5])
lags = 5
g = TimeseriesGenerator(a, a, length=lags, batch_size=5)

model = Sequential()
model.add(SimpleRNN(500, activation='relu', input_shape=(lags, 1)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

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

### 输出图像，图
plt.figure(figsize=(10, 6))
plt.plot(x[lags:], d[lags:], label='data', alpha=0.75)
plt.plot(x[lags:], y, 'r.', label='pred', ms=3)
plt.axvline(-2 * np.pi, c='g', ls='--')
plt.axvline(2 * np.pi, c='g', ls='--')
plt.text(-15, 22, 'out-of-sample')
plt.text(-2, 22, 'in-sample')
plt.text(10, 22, 'out-of-sample')
plt.legend()


#########################8.3金融价格序列
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

raw = pd.read_csv('aiif_eikon_id_eur_usd（第7章）.csv', index_col=0, parse_dates=True)
symbol = 'EUR_USD'

###导入数据并重新采样
def generate_data():
    data = pd.DataFrame(raw['CLOSE'])#选择1列
    data.columns = [symbol]#将该列重命名
    #重新采样数据
    data = data.resample('30min', label='right').last().ffill()
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

model = create_rnn_model()

model.fit(g, epochs=500, steps_per_epoch=10,
          verbose=False)

###样本内预测
y = model.predict(g, verbose=False) #2维数组
print(len(g))
print(len(y)) #生成g是batch_size=5,每一个批次5个样本，len(y)=5*len(g)

data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()#y是2维数组，需要打平为1维

# 绘制预测结果，图
plt.figure(figsize=(14, 7))  # 增加图像尺寸
data[[symbol, 'pred']].plot(
    figsize=(14, 7), style=['b', 'r-.'], alpha=0.75, linewidth=1)  # 降低线条宽度
plt.title('Financial Price Series Prediction')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Price')  # Y轴标签
plt.legend(['Actual Values', 'Predicted Values'])  # 图例
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()

# 放大视图，图
plt.figure(figsize=(14, 7))  # 增加图像尺寸
data[[symbol, 'pred']].iloc[50:100].plot(
    figsize=(14, 7), style=['b', 'r-.'], alpha=0.75, linewidth=1)  # 降低线条宽度
plt.title('Zoomed View: Time Steps 50 to 100')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Price')  # Y轴标签
plt.legend(['Actual Values', 'Predicted Values'])  # 图例
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()


### 计算并打印回归性能指标
actual = data[symbol].iloc[lags:]
predicted = data['pred'].iloc[lags:]

# 计算指标
mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print("8.3")
print(f"均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 计算方向准确度
actual_direction = np.sign(actual)
predicted_direction = np.sign(predicted)
direction_accuracy = accuracy_score(actual_direction, predicted_direction)

print(f"方向准确度: {direction_accuracy:.4f}")



##########################8.4金融收益率序列
data = generate_data()
data['r'] = np.log(data / data.shift(1))
data.dropna(inplace=True)
data = (data - data.mean()) / data.std()
r = data['r'].values
r = r.reshape((len(r), -1))

g = TimeseriesGenerator(r, r, length=lags, batch_size=5)
model = create_rnn_model()

model.fit(g, epochs=500, steps_per_epoch=10,
          verbose=False)

y = model.predict(g, verbose=False)
print("8.4")
print(f"生成器中的批次数量: {len(g)}")
print(f"预测值的数量: {len(y)}")  # 生成g是batch_size=5, 每个批次5个样本，len(y)=5*len(g)

data['pred'] = np.nan
data['pred'].iloc[lags:] = y.flatten()
data.dropna(inplace=True)

# 绘制完整的实际值与预测值：图
plt.figure(figsize=(14, 7))  # 增加图像尺寸
data[['r', 'pred']].plot(
    figsize=(14, 7), style=['b', 'r-.'], alpha=0.75, linewidth=1)  # 降低线条宽度
plt.title('Financial Return Series Prediction')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签
plt.legend(['Actual Values', 'Predicted Values'])  # 图例
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()

# 绘制放大视图：图
plt.figure(figsize=(14, 7))  # 增加图像尺寸
data[['r', 'pred']].iloc[50:100].plot(
    figsize=(14, 7), style=['b', 'r-.'], alpha=0.75, linewidth=1)  # 降低线条宽度
plt.title('Zoomed View: Financial Return Prediction from Time Step 50 to 100')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签
plt.legend(['Actual Values', 'Predicted Values'])  # 图例
plt.axhline(0, c='grey', ls='--')  # 添加水平线表示零收益率
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()


# 计算并打印方向预测准确度
actual_direction = np.sign(data['r'])
predicted_direction = np.sign(data['pred'])
direction_accuracy = accuracy_score(actual_direction, predicted_direction)
print(f"样本内方向预测准确度: {direction_accuracy:.4f}")  # 准确度

### 样本外预测
split = int(len(r) * 0.8)
train = r[:split]
test = r[split:]
g_train = TimeseriesGenerator(train, train, length=lags, batch_size=5)

set_seeds()
model = create_rnn_model(hu=100)

model.fit(g_train, epochs=100, steps_per_epoch=10, verbose=False)  # 训练集拟合模型

g_test = TimeseriesGenerator(test, test, length=lags, batch_size=5)  # 测试集
y_test = model.predict(g_test)
predicted_test = np.sign(y_test).flatten()
actual_test = np.sign(test[lags:])
test_direction_accuracy = accuracy_score(actual_test, predicted_test)
print(f"样本外方向预测准确度: {test_direction_accuracy:.4f}")  # 准确度



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

g = TimeseriesGenerator(train.values, train['r'].values,
                        length=lags, batch_size=5)

print("训练集收益率样本前10个值：", train['r'].values[:10])

# 创建和训练RNN模型
set_seeds()
model = create_rnn_model(hu=100, features=len(data.columns),
                         layer='SimpleRNN')

model.fit(g, epochs=100, steps_per_epoch=10,
          verbose=False)

# 创建测试集生成器
g_test = TimeseriesGenerator(test.values, test['r'].values,
                             length=lags, batch_size=5)

# 进行预测
y_pred = model.predict(g_test).flatten()

# 将预测值赋值到测试数据中
# 由于 TimeseriesGenerator 会跳过前 lags 个数据点，因此需要对齐预测结果
test = test.iloc[lags:].copy()  # 跳过前 lags 个数据点
test['pred'] = y_pred

# 绘制整个测试集的预测结果
plt.figure(figsize=(14, 7))  # 增加图像宽度以更好地显示整个数据
# 绘制实际值
plt.plot(test.index, test['r'], label='Actual Returns', color='b')
# 绘制预测值
plt.plot(test.index, test['pred'], label='Predicted Returns', color='r', linestyle='--')
plt.title('Financial Return Prediction')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Normalized Returns')  # Y轴标签
plt.legend(['Actual Returns', 'Predicted Returns'])  # 图例
plt.axhline(0, color='grey', linestyle='--')  # 添加水平线表示零收益率
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()


# 计算并打印回归性能指标
actual = test['r'].values
predicted = test['pred'].values

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predicted)

print("8.5.1")
print(f"均绝对误差 (MAE): {mae:.4f}")
print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"决定系数 (R²): {r2:.4f}")

# 计算并打印方向预测准确度
direction_accuracy = accuracy_score(np.sign(test['r'].values), np.sign(test['pred'].values))
print(f"方向预测准确度: {direction_accuracy:.4f}")  # 准确度



####8.5.2分类
set_seeds()
model = create_rnn_model(hu=50,
            features=len(data.columns),
            layer='LSTM',
            algorithm='classification')

train_y = np.where(train['r'] > 0, 1, 0)
print(train_y)
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

# 创建自定义生成器
g = CustomTimeseriesGenerator(train.values, train_y, length=lags, batch_size=5, class_weights=class_weights)

# 训练模型时不再需要传递 class_weight
model.fit(g, epochs=5, steps_per_epoch=10, verbose=False)

# g = TimeseriesGenerator(train.values, train_y,
#                         length=lags, batch_size=5)

# model.fit(g, epochs=5, steps_per_epoch=10,
#           verbose=False, class_weight=cw(train_y))
## 改动结束

test_y = np.where(test['r'] > 0, 1, 0)

g_ = TimeseriesGenerator(test.values, test_y,
                         length=lags, batch_size=5)

y = np.where(model.predict(g_, batch_size=None) > 0.5,
             1, 0).flatten()

np.bincount(y)

print(accuracy_score(test_y[lags:], y))  ### 准确度

###市场方向预测表现更好


###8.5.3深度RNN
from tensorflow.keras.layers import Dropout

def create_deep_rnn_model(hl=2, hu=100, layer='SimpleRNN',
                          optimizer='rmsprop', features=1,
                          dropout=False, rate=0.3, seed=100):
    # 保证最少2个隐藏层
    if hl < 2:
        hl = 2
    if layer == 'SimpleRNN':
        layer_type = SimpleRNN
    else:
        layer_type = LSTM
    model = Sequential()
    # 第一个隐藏层
    model.add(layer_type(hu, input_shape=(lags, features), return_sequences=True))
    if dropout:
        model.add(Dropout(rate, seed=seed))
    # 中间隐藏层
    for _ in range(2, hl):
        model.add(layer_type(hu, return_sequences=True))
        if dropout:
            model.add(Dropout(rate, seed=seed))  # Dropout层
    # 最终隐藏层（不返回序列）
    model.add(layer_type(hu))
    model.add(Dense(1, activation='sigmoid'))  # 建立分类模型
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 设置随机种子
set_seeds()

# 创建深度RNN模型
model = create_deep_rnn_model(
            hl=2, hu=50, layer='SimpleRNN',
            features=len(data.columns),
            dropout=True, rate=0.3)

# 显示模型摘要
model.summary()

# 训练模型
model.fit(g, epochs=200, steps_per_epoch=10, verbose=False)

# 进行预测
y_pred_probs_deep = model.predict(g_, batch_size=None)
y_pred_deep = (y_pred_probs_deep > 0.5).astype(int).flatten()

print("8.5.3")
# 打印预测类别的分布
print("深度RNN预测类别分布：", np.bincount(y_pred_deep))

# 计算并打印方向预测准确度
accuracy_deep = accuracy_score(test_y[lags:], y_pred_deep)
print(f"深度RNN方向预测准确度: {accuracy_deep:.4f}")  # 准确度

# 将预测结果对齐到测试集
test_aligned_deep = test.iloc[lags:].copy()
test_aligned_deep['pred_deep'] = y_pred_deep

# 绘制整个测试集的深度RNN预测结果
plt.figure(figsize=(14, 7))  # 增加图像宽度以更好地显示整个数据
# 绘制实际方向
plt.plot(test_aligned_deep.index, test_aligned_deep['r'], label='Actual Direction', color='b')
# 绘制深度RNN预测方向
plt.plot(test_aligned_deep.index, test_aligned_deep['pred_deep'], label='Deep RNN Predicted Direction', color='r', linestyle='--')
plt.title('Deep RNN Financial Return Direction Prediction')  # 图表标题
plt.xlabel('Time Steps')  # X轴标签
plt.ylabel('Direction')  # Y轴标签
plt.legend(['Actual Direction', 'Deep RNN Predicted Direction'])  # 图例
plt.axhline(0.5, color='grey', linestyle='--')  # 添加中线，表示阈值
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线以提升可读性
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()


# 打印分类报告
#print("深度RNN分类报告：\n", classification_report(test_y[lags:], y_pred_deep))

