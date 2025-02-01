# -*- coding: utf-8 -*-

### 输出警告改动：屏蔽警告 
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 改动：只显示错误信息 

############################# 期末实验作业
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from pylab import plt, mpl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision = 4
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'


# 加载数据集
file_path = 'TRD_Dalyr_Final.csv'  # 数据文件名
data = pd.read_csv(file_path)

# 查看数据结构
print("数据集基本信息：")
print(data.info())
print("\n数据集前五行：")
print(data.head())

# 数据预处理
data['Trddt'] = pd.to_datetime(data['Trddt']) # 将日期列转换为日期格式
data.set_index('Trddt', inplace=True) # 设置日期列为索引
data2 = data.copy()
# 选择收盘价列并重采样为每周数据
weekly_data = pd.DataFrame(data['Clsprc'].resample('W', label='right').last().ffill())
weekly_data.columns = ['Weekly Close'] 




###（1）画出股票全样本周度收盘价时间序列图
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.plot(weekly_data, label='Weekly Close Price')  # 绘制折线图
plt.title('Weekly Close Price Time Series')  # 图表标题
plt.xlabel('Date')  # X轴标签
plt.ylabel('Price')  # Y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 添加网格
plt.show()




###（2）构造3个特征（额外探索：实际多于3个，为交易价格、成交量、成交金额3类的多个特征）
def add_features(df, window=30):
    """
    增加交易价格、成交量和成交金额的多种特征，使用时按要求挑3个即可
    """
    # 第一类：交易价格信息特征
    df['r'] = np.log(df['Clsprc'] / df['Clsprc'].shift(1))  # 特征0：对数收益率
    df['mom'] = df['r'].rolling(window).mean()  # 特征1：对数收益率动量
    df['vol'] = df['r'].rolling(window).std()  # 特征2：滚动波动率
    df['d'] = np.where(df['r'] > 0, 1, 0)  # 特征3：当日收益率方向（二进制0、1特征）
    df['ewm'] = df['Clsprc'].ewm(alpha=0.5).mean()  # 特征4：指数加权移动平均
    df['daily_range'] = df['Hiprc'] - df['Loprc']  # 特征5：日内波动幅度
    df['daily_return'] = (df['Clsprc'] / df['Opnprc']) - 1  # 特征6：日内涨跌幅

    # 第二类：成交量信息特征
    df['volume_sma'] = df['Dnshrtrd'].rolling(window).mean()  # 特征7：成交量的简单移动平均线
    df['volume_min'] = df['Dnshrtrd'].rolling(window).min()  # 特征8：成交量的滚动最小值

    # 第三类：成交金额信息特征
    df['turnover_vol'] = df['Dnvaltrd'].rolling(window).std()  # 特征9：成交金额的滚动波动率
    df['turnover_max'] = df['Dnvaltrd'].rolling(window).max()  # 特征10：成交金额的滚动最大值

    # 清除NAN值
    df.dropna(inplace=True)

    return df

# 调用函数
data = add_features(data, window=30)

# 检查构造的特征
print("\n构造的特征前五行：")
print(data[['r', 'mom', 'vol', 'd', 'ewm', 'daily_range', 'daily_return', 
           'volume_sma', 'volume_min', 'turnover_vol', 'turnover_max']].head())

# 绘制特征的时间序列图
features = ['r', 'mom', 'vol', 'd', 'ewm', 'daily_range', 'daily_return', 
            'volume_sma', 'volume_min', 'turnover_vol', 'turnover_max']
titles = ['Log Return (Feature 0)', 'Momentum (Feature 1)', 'Volatility (Feature 2)', 
          'Daily Return Direction (Feature 3)', 'EWM (Feature 4)', 'Daily Range (Feature 5)', 
          'Daily Return (Feature 6)', 'Volume SMA (Feature 7)', 'Volume Min (Feature 8)', 
          'Turnover Volatility (Feature 9)', 'Turnover Max (Feature 10)']

# 绘制每个特征的时间序列图（日度数据）
for i, feature in enumerate(features):
    plt.figure(figsize=(12, 6))
    plt.plot(data[feature], label=titles[i])
    plt.title(titles[i])
    plt.xlabel('Date')
    plt.ylabel(feature.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.show()

    
    
    
###（3）基于日度收益信息和构造的3个特征，使用OLS、密集神经网络、循环神经网络分别对市场弱式效率进行检验
lags = 7  # 7天滞后
def add_lags(df,lags):
    '''增加滞后项'''
    cols = []
    # 按要求:日度收益信息（基本的收盘价Clsprc和收益率r）
    # 外加选用3个特征：动量mom，收益滚动波动率vol, 成交量简单移动平均线volume_sma
    features = ['Clsprc', 'r', 'mom', 'vol', 'volume_sma']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols

# 生成滞后特征
data, cols = add_lags(data, lags)


## 1.OLS回归
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
print("\n--- OLS 回归 ---")
# 使用OLS对市场弱式效率进行检验
a = 0
reg = np.linalg.lstsq(data[cols], data['r'], rcond=-1)[a]  # 线性回归求解
pred = np.dot(data[cols], reg)  # 使用OLS模型预测

# 计算回归任务的评估指标
def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, mae, r2

acc_ols = accuracy_score(np.sign(data['r']), np.sign(pred))  # 准确率分析
mse_ols, mae_ols, r2_ols = evaluate_regression(data['r'], pred) # 计算OLS的回归指标

# 打印OLS的指标
print(f"OLS Accuracy: {acc_ols:.4f}")
print(f"OLS MSE: {mse_ols:.4f}")
print(f"OLS MAE: {mae_ols:.4f}")
print(f"OLS R²: {r2_ols:.4f}")


## 2.密集神经网络（DNN）
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
print("\n--- 密集神经网络 (DNN) ---")
# 定义DNN模型
optimizer = Adam(learning_rate=0.001) #默认优化器
def create_dnn_model(input_dim, hl=1, hu=128, optimizer=optimizer):
    model = Sequential()
    model.add(Dense(hu, input_dim=input_dim, activation='relu'))  # 第一层
    for _ in range(hl):  # 隐藏层
        model.add(Dense(hu, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 输出层
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def set_seeds(seed=100):
    random.seed(seed) #设定Python随机数种子
    np.random.seed(seed) #设定Numpy随机数种子
    tf.random.set_seed(seed) #设定TensorFlow随机数种子 

# 数据集拆分为训练集和测试集
split = int(len(data) * 0.8)  # 80% 训练集，20% 测试集
train = data.iloc[:split].copy()
test = data.iloc[split:].copy()

# 类别权重计算（处理类别不平衡）
def compute_class_weights(df):
    c0, c1 = np.bincount(df['d'])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}

# 创建并训练DNN模型
set_seeds()
model_dnn = create_dnn_model(input_dim=len(cols), hl=1, hu=128)
history = model_dnn.fit(
    train[cols], train['d'], epochs=50, verbose=False, validation_split=0.2, shuffle=False,
    class_weight=compute_class_weights(train)
)

# 评估DNN模型性能
y_pred_proba_dnn = model_dnn.predict(test[cols]).flatten()  # 获取预测概率
y_pred_dnn = (y_pred_proba_dnn >= 0.5).astype(int)          # 转换为类别标签

# 计算准确率、精确率、召回率、F1 分数
acc_dnn = accuracy_score(test['d'], y_pred_dnn)
precision_dnn = precision_score(test['d'], y_pred_dnn)
recall_dnn = recall_score(test['d'], y_pred_dnn)
f1_dnn = f1_score(test['d'], y_pred_dnn)

# 计算 AUC-ROC 曲线
fpr_dnn, tpr_dnn, thresholds_dnn = roc_curve(test['d'], y_pred_proba_dnn)
auc_dnn = auc(fpr_dnn, tpr_dnn)

# 打印评估指标
print(f"DNN Accuracy: {acc_dnn:.4f}")
print(f"DNN Precision: {precision_dnn:.4f}")
print(f"DNN Recall: {recall_dnn:.4f}")
print(f"DNN F1-Score: {f1_dnn:.4f}")
print(f"DNN AUC: {auc_dnn:.4f}")

# 绘制 AUC-ROC 曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {auc_dnn:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')  # 对角线表示随机分类器
plt.title('DNN AUC-ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


## 3.循环神经网络（RNN/LSTM）
print("\n--- 循环神经网络 (RNN/LSTM) ---")
# 定义RNN/LSTM模型
def create_rnn_model(hu=100, lags=5, features=1, layer='SimpleRNN', algorithm='estimation'):
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

# 准备RNN/LSTM训练数据
g_train = TimeseriesGenerator(train[cols].values, train['r'].values, length=lags, batch_size=5)
g_test = TimeseriesGenerator(test[cols].values, test['r'].values, length=lags, batch_size=5)

# 创建并训练SimpleRNN和LSTM模型
set_seeds()
model_rnn = create_rnn_model(hu=100, lags=lags, features=len(cols), layer='SimpleRNN')
set_seeds()
model_lstm = create_rnn_model(hu=100, lags=lags, features=len(cols), layer='LSTM')
model_rnn.fit(g_train, epochs=100, steps_per_epoch=10, verbose=False)
model_lstm.fit(g_train, epochs=100, steps_per_epoch=10, verbose=False)

# 评估RNN模型性能
pred_rnn = model_rnn.predict(g_test).flatten()
pred_lstm = model_lstm.predict(g_test).flatten()
acc_rnn = accuracy_score(np.sign(test['r'].iloc[lags:]), np.sign(pred_rnn)) # 准确率分析
acc_lstm = accuracy_score(np.sign(test['r'].iloc[lags:]), np.sign(pred_lstm))

# 打印RNN模型的评估指标
print(f"SimpleRNN Accuracy: {acc_rnn:.4f}")
print(f"LSTM Accuracy: {acc_lstm:.4f}")




###（4） 使用密集神经网络对股票价格日度收益进行预测，并根据预测构建投资策略
# 定义无风险收益率（年化利率）
risk_free_rate = 0.01  # 无风险年利率 1%
Daily_rate = risk_free_rate / 365 # 无风险日利率

# 重新处理数据，使用日度行情信息和构造的3个特征
data2 = add_features(data2, window=30)

lags = 7  # 7天滞后
def add_lags_2(df2,lags):
    '''增加滞后项'''
    cols2 = []
    # 按要求:日度行情信息（所有数据列以及重要基本信息）
    # 外加选用3个特征：动量mom，收益滚动波动率vol, 成交量简单移动平均线volume_sma
    features2 = ['Opnprc', 'Hiprc', 'Loprc', 'Clsprc', 'Dnshrtrd', 'Dnvaltrd', 'r', 'd',
                 'mom', 'vol', 'volume_sma']
    for f2 in features2:
        for lag in range(1, lags + 1):
            col2 = f'{f2}_lag_{lag}'
            df2[col2] = df2[f2].shift(lag)
            cols2.append(col2)
    df2.dropna(inplace=True)
    return df2, cols2

# 生成滞后特征
data2, cols2 = add_lags_2(data2, lags)

# 数据集切分为训练集和测试集，连续切分了。因为是时间序列所以随机切分不太合适？
split = int(len(data2) * 0.8)  # 80% 训练集，20% 测试集
train = data2.iloc[:split].copy()  # 从 data2 中切分出 train
test = data2.iloc[split:].copy()  # 从 data2 中切分出 test

# 创建并训练DNN模型，训练测试集8:2划分
set_seeds()
optimizer = Adam(learning_rate=0.0001)
model_dnn = create_dnn_model(input_dim=len(cols2), hl=1, hu=128, optimizer=optimizer)
history = model_dnn.fit(
    train[cols2], train['d'], epochs=100, verbose=False, validation_split=0.2, shuffle=False
)

# 获取预测结果
y_pred_dnn = model_dnn.predict(test[cols2]).flatten()

# 构建投资策略1（DNN版）：如果预测收益为正就投股票，否则投无风险
portfolio_value = 1  # 初始组合价值
daily_returns = []  # 存储每日收益率

for i, pred_direction in enumerate(y_pred_dnn):
    if pred_direction == 1:  # 如果预测结果为上涨方向
        portfolio_value *= (1 + test['r'].iloc[i])  # 投资于股票
        daily_returns.append(test['r'].iloc[i])
    else:  # 如果预测结果为下跌方向
        portfolio_value *= (1 + Daily_rate)  # 投资于无风险资产
        daily_returns.append(Daily_rate)

# 评估模型性能：准确率
test_labels = (test['r'] > 0).astype(int)  # 将真实收益率转换为方向（涨=1, 跌=0）
accuracy = (test_labels == y_pred_dnn).mean()  # 计算方向准确率
print(f"模型预测的方向准确率: {accuracy:.4f}")

# 计算投资策略表现
portfolio_return = portfolio_value - 1  # 投资策略总收益率
daily_returns = np.array(daily_returns)

# 读数据沪深300指数
index_data = pd.read_csv('CSI300_Daily.csv', index_col='Date', parse_dates=True)  # 沪深300文件
index_data['Close'] = index_data['Close'].str.replace(',', '').astype(float)  # 转换为浮点数
index_data = index_data.sort_index()  # 按日期升序排列
index_data['r'] = np.log(index_data['Close'] / index_data['Close'].shift(1)).dropna() # 收益率
index_data.dropna(inplace=True)  # 删除 NaN 行
# 确保 daily_returns 和 index_data['r'] 时间对齐
daily_returns = daily_returns[-len(index_data):]  # 截取到 index_data 的长度
index_returns = index_data['r'].iloc[-len(daily_returns):]  # 对齐时间

# 计算贝塔值
covariance = np.cov(daily_returns, index_data['r'].iloc[-len(daily_returns):])[0, 1] # 协方差
market_variance = np.var(index_data['r'].iloc[-len(daily_returns):]) # 市场指数的方差
beta = covariance / market_variance # 贝塔值

# 计算夏普比率
sharpe_ratio = (daily_returns.mean() - Daily_rate) / daily_returns.std()

# 输出结果
print(f"投资策略总收益率: {portfolio_return:.4f}")
print(f"投资策略贝塔值: {beta:.4f}")
print(f"投资策略夏普比率: {sharpe_ratio:.4f}")


## 额外探索：OLS 方法预测收益率，并根据预测构建投资策略
# 使用OLS对收益率进行预测
from sklearn.linear_model import LinearRegression

# 定义 OLS 模型
ols_model = LinearRegression()

# 使用训练集进行训练
ols_model.fit(train[cols2], train['r'])

# 在测试集上进行预测
ols_pred = ols_model.predict(test[cols2])
ols_pred_class = (ols_pred > 0).astype(int)  # 将预测的收益率转换为方向（涨=1，跌=0）

# 构建投资策略2（OLS版）：更精准地比较预测的收益率与无风险收益，如果更大才投股票
ols_portfolio_value = 1  # 初始组合价值
ols_daily_returns = []  # 存储每日收益率

for i, pred_return in enumerate(ols_pred):
    if pred_return > Daily_rate:  # 如果预测收益率高于无风险收益率（日化），买入股票
        ols_portfolio_value *= (1 + test['r'].iloc[i])
        ols_daily_returns.append(test['r'].iloc[i])
    else:  # 否则投资无风险收益
        ols_portfolio_value *= (1 + Daily_rate)
        ols_daily_returns.append(Daily_rate)

# 评估模型性能：准确率
ols_accuracy = (test_labels == ols_pred_class).mean()  # 计算方向准确率
print(f"OLS 模型预测的方向准确率: {ols_accuracy:.4f}")

# 计算投资策略表现
ols_portfolio_return = ols_portfolio_value - 1  # 投资策略总收益率
ols_daily_returns = np.array(ols_daily_returns)

# 确保 daily_returns 和 index_data['r'] 时间对齐
ols_daily_returns = ols_daily_returns[-len(index_data):]  # 截取到 index_data 的长度
ols_index_returns = index_data['r'].iloc[-len(ols_daily_returns):]  # 对齐时间

# 计算 OLS 投资策略的贝塔值
ols_covariance = np.cov(ols_daily_returns, ols_index_returns)[0, 1]  # 协方差
ols_market_variance = np.var(ols_index_returns)  # 市场指数的方差
ols_beta = ols_covariance / ols_market_variance  # 贝塔值

# 计算 OLS 投资策略的夏普比率
ols_sharpe_ratio = (ols_daily_returns.mean() - Daily_rate) / ols_daily_returns.std()

# 输出 OLS 结果
print(f"OLS 投资策略总收益率: {ols_portfolio_return:.4f}")
print(f"OLS 投资策略贝塔值: {ols_beta:.4f}")
print(f"OLS 投资策略夏普比率: {ols_sharpe_ratio:.4f}")




###（5）验证归一化和正则化对问题（4）的影响
from tensorflow.keras.regularizers import l1, l2

# 投资策略及评估函数
def evaluate_model_performance(y_pred, test_r, daily_rate, index_r):
    """
    生成投资策略并评估模型预测性能，包括方向准确率、收益率、贝塔值和夏普比率。
    """
    portfolio_value = 1  # 初始组合价值
    daily_returns = []  # 存储每日收益率

    # 投资策略逻辑
    for i, pred_direction in enumerate(y_pred):
        if pred_direction == 1:  # 如果预测结果为上涨方向
            portfolio_value *= (1 + test['r'].iloc[i])  # 投资于股票
            daily_returns.append(test['r'].iloc[i])
        else:  # 如果预测结果为下跌方向
            portfolio_value *= (1 + Daily_rate)  # 投资于无风险资产
            daily_returns.append(Daily_rate)

    portfolio_return = portfolio_value - 1  # 总收益率
    daily_returns = np.array(daily_returns)

    # 方向准确率
    test_labels = (test_r > 0).astype(int)
    accuracy = (test_labels == y_pred).mean()

    # 贝塔值计算
    covariance = np.cov(daily_returns, index_r)[0, 1]
    market_variance = np.var(index_r)
    beta = covariance / market_variance

    # 夏普比率计算
    sharpe_ratio = (daily_returns.mean() - daily_rate) / daily_returns.std()

    return accuracy, portfolio_return, beta, sharpe_ratio


# 定义包含正则化的 DNN 模型
def create_dnn_with_regularization(input_dim, hl=1, hu=128, reg_type='l2', reg_value=0.001):
    if reg_type == 'l1':
        regularizer = l1(reg_value)
    elif reg_type == 'l2':
        regularizer = l2(reg_value)
    else:
        regularizer = None
    model = Sequential()
    model.add(Dense(hu, input_dim=input_dim, activation='relu', kernel_regularizer=regularizer))
    for _ in range(hl):
        model.add(Dense(hu, activation='relu', kernel_regularizer=regularizer))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model


## 1. 归一化影响分析
# 归一化数据
mu, std = train[cols2].mean(), train[cols2].std()
train_normalized = (train[cols2] - mu) / std
test_normalized = (test[cols2] - mu) / std

# 使用归一化数据训练 DNN
set_seeds()
optimizer = Adam(learning_rate=0.001)
model_dnn = create_dnn_model(input_dim=len(cols2), hl=1, hu=128, optimizer=optimizer)
history = model_dnn.fit(
    train_normalized, train['d'], epochs=50, verbose=False, validation_split=0.2, shuffle=False
)
y_pred_dnn = model_dnn.predict(test_normalized).flatten()
y_pred_class = (y_pred_dnn > 0.5).astype(int)

# 评估模型性能
accuracy, portfolio_return, beta, sharpe_ratio = evaluate_model_performance(
    y_pred_class, test['r'], Daily_rate, index_data['r'].iloc[-len(y_pred_dnn):]
)

print(f"归一化后投资策略方向准确率: {accuracy:.4f}")
print(f"归一化后投资策略总收益率: {portfolio_return:.4f}")
print(f"归一化后投资策略贝塔值: {beta:.4f}")
print(f"归一化后投资策略夏普比率: {sharpe_ratio:.4f}")

# 绘制训练和验证准确率曲线
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)  # 创建1行2列的子图，第一幅子图
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Normalized DNN: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1, 2, 2)  # 创建1行2列的子图，第二幅子图
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Normalized DNN: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## 2. 正则化影响分析
set_seeds()
model_dnn_reg = create_dnn_with_regularization(
    input_dim=len(cols2), reg_type='l2', reg_value=0.001)
history_reg = model_dnn_reg.fit(
    train[cols2], train['d'], epochs=100, verbose=False, validation_split=0.2, shuffle=False
)
y_pred_dnn_reg = model_dnn_reg.predict(test[cols2]).flatten()

# 评估模型性能
accuracy, portfolio_return, beta, sharpe_ratio = evaluate_model_performance(
    y_pred_dnn_reg, test['r'], Daily_rate, index_data['r'].iloc[-len(y_pred_dnn_reg):]
)

print(f"正则化后投资策略方向准确率: {accuracy:.4f}")
print(f"正则化后投资策略总收益率: {portfolio_return:.4f}")
print(f"正则化后投资策略贝塔值: {beta:.4f}")
print(f"正则化后投资策略夏普比率: {sharpe_ratio:.4f}")

# 绘制正则化模型的训练和验证曲线
plt.figure(figsize=(16, 6))
# 准确率曲线
plt.subplot(1, 2, 1)  # 创建1行2列的子图，第一幅子图
plt.plot(history_reg.history['accuracy'], label='Training Accuracy')
plt.plot(history_reg.history['val_accuracy'], label='Validation Accuracy')
plt.title('Regularization: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1, 2, 2)  # 第二幅子图
plt.plot(history_reg.history['loss'], label='Training Loss')
plt.plot(history_reg.history['val_loss'], label='Validation Loss')
plt.title('Regularization: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## 3. 归一化+正则化影响分析
set_seeds()
model_norm_reg = create_dnn_with_regularization(
    input_dim=len(cols2), reg_type='l2', reg_value=0.001)  
history_norm_reg = model_norm_reg.fit(
    train_normalized, train['d'], epochs=50, verbose=False, validation_split=0.2, shuffle=False
)
y_pred_norm_reg = model_norm_reg.predict(test_normalized).flatten()
y_pred_class_norm_reg = (y_pred_norm_reg > 0.5).astype(int)

# 评估模型性能
accuracy, portfolio_return, beta, sharpe_ratio = evaluate_model_performance(
    y_pred_class_norm_reg, test['r'], Daily_rate, index_data['r'].iloc[-len(y_pred_norm_reg):]
)

print(f"归一化+正则化后投资策略方向准确率: {accuracy:.4f}")
print(f"归一化+正则化后投资策略总收益率: {portfolio_return:.4f}")
print(f"归一化+正则化后投资策略贝塔值: {beta:.4f}")
print(f"归一化+正则化后投资策略夏普比率: {sharpe_ratio:.4f}")

# 绘制归一化+正则化模型的训练和验证曲线
plt.figure(figsize=(16, 6))
# 准确率曲线
plt.subplot(1, 2, 1)  # 创建1行2列的子图，第一幅子图
plt.plot(history_norm_reg.history['accuracy'], label='Training Accuracy')
plt.plot(history_norm_reg.history['val_accuracy'], label='Validation Accuracy')
plt.title('Normalization + Regularization: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1, 2, 2)  # 第二幅子图
plt.plot(history_norm_reg.history['loss'], label='Training Loss')
plt.plot(history_norm_reg.history['val_loss'], label='Validation Loss')
plt.title('Normalization + Regularization: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## 额外探索：4. 归一化+暂退影响分析
set_seeds()

# 创建具有 Dropout 的 DNN 模型
optimizer = Adam(learning_rate=0.001) #默认优化器
def create_dnn_dropout(input_dim, dropout_rate=0.5, hl=1, hu=128, optimizer=optimizer):
    model = Sequential()
    # 输入层 + 隐藏层
    model.add(Dense(hu, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout_rate))  # 添加 Dropout
    # 额外的隐藏层
    for _ in range(hl - 1):
        model.add(Dense(hu, activation='relu'))
        model.add(Dropout(dropout_rate))  # 在每一层添加 Dropout
    # 输出层
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 定义模型
model_dnn_dropout = create_dnn_dropout(
    input_dim=len(cols2), dropout_rate=0.3, hl=1, hu=128, optimizer=optimizer)

# 使用归一化数据训练
history_norm_dropout = model_dnn_dropout.fit(
    train_normalized, train['d'], epochs=50, verbose=False, validation_split=0.2, shuffle=False
)

# 预测结果
y_pred_norm_drop = model_dnn_dropout.predict(test_normalized).flatten()
y_pred_class_drop = (y_pred_norm_drop > 0.5).astype(int)

# 评估模型性能
accuracy, portfolio_return, beta, sharpe_ratio = evaluate_model_performance(
    y_pred_class_drop, test['r'], Daily_rate, index_data['r'].iloc[-len(y_pred_norm_drop):]
)

print(f"归一化+暂退后投资策略方向准确率: {accuracy:.4f}")
print(f"归一化+暂退后投资策略总收益率: {portfolio_return:.4f}")
print(f"归一化+暂退后投资策略贝塔值: {beta:.4f}")
print(f"归一化+暂退后投资策略夏普比率: {sharpe_ratio:.4f}")

# 绘制训练和验证准确率曲线
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)  # 创建1行2列的子图，第一幅子图
plt.plot(history_norm_dropout.history['accuracy'], label='Training Accuracy')
plt.plot(history_norm_dropout.history['val_accuracy'], label='Validation Accuracy')
plt.title('Normalization + Dropout: Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
# 损失曲线
plt.subplot(1, 2, 2)  # 创建1行2列的子图，第二幅子图
plt.plot(history_norm_dropout.history['loss'], label='Training Loss')
plt.plot(history_norm_dropout.history['val_loss'], label='Validation Loss')
plt.title('Normalization + Dropout: Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## 额外探索：5. 优化器选择影响分析
# 定义优化器列表
optimizers = ['rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

# 循环测试不同优化器
for opt in optimizers:
    set_seeds()
    model_dnn_opt = create_dnn_model(input_dim=len(cols2), hl=1, hu=128, optimizer=opt)
    history_opt = model_dnn_opt.fit(
        train[cols2], train['d'], epochs=50, verbose=False, validation_split=0.2, shuffle=False
    )
    y_pred_dnn_opt = model_dnn_opt.predict(test[cols2]).flatten()

    # 评估模型性能
    accuracy, portfolio_return, beta, sharpe_ratio = evaluate_model_performance(
        y_pred_dnn_opt, test['r'], Daily_rate, index_data['r'].iloc[-len(y_pred_dnn_opt):]
    )

    print(f"使用优化器 {opt} 的模型预测方向准确率: {accuracy:.4f}")
    print(f"使用优化器 {opt} 的模型预测总收益率: {portfolio_return:.4f}")
    print(f"使用优化器 {opt} 的模型预测贝塔值: {beta:.4f}")
    print(f"使用优化器 {opt} 的模型预测夏普比率: {sharpe_ratio:.4f}")




###（6）说明使用深度强化学习改进投资策略业绩表现的方法
# 详细思路见报告，大致代码思路如下（并不运行实现）

# from collections import deque
# # 参数定义
# STATE_SIZE = len(cols2)  # 状态空间大小（输入特征数量）
# ACTION_SIZE = 10  # 动作空间大小（分为10个离散权重，比如0%, 10%, ..., 100%）
# GAMMA = 0.95  # 折扣因子
# LEARNING_RATE = 0.001  # 学习率
# MEMORY_SIZE = 2000  # 经验回放缓冲区大小
# BATCH_SIZE = 64  # 每次训练采样大小

# # 构建深度 Q 网络（DQN）
# def create_dqn_model(state_size, action_size):
#     model = Sequential()
#     model.add(Dense(128, input_dim=state_size, activation='relu'))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(action_size, activation='linear'))  # 输出 Q 值
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
#                   loss='mse')
#     return model

# # 定义 DQN 智能体
# class DQNAgent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.memory = deque(maxlen=MEMORY_SIZE)  # 经验回放缓冲区
#         self.model = create_dqn_model(state_size, action_size)  # 主网络
#         self.target_model = create_dqn_model(state_size, action_size)  # 目标网络
#         self.epsilon = 1.0  # 初始探索率
#         self.epsilon_decay = 0.995  # 探索率衰减
#         self.epsilon_min = 0.01  # 最小探索率

#     # 保存经验
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     # 选择动作（ε-贪婪策略）
#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)  # 随机选择动作
#         q_values = self.model.predict(state, verbose=0)
#         return np.argmax(q_values[0])  # 选择 Q 值最大的动作

#     # 训练主网络
#     def replay(self, batch_size):
#         if len(self.memory) < batch_size:
#             return
#         minibatch = random.sample(self.memory, batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = self.model.predict(state, verbose=0)
#             if done:
#                 target[0][action] = reward
#             else:
#                 t = self.target_model.predict(next_state, verbose=0)
#                 target[0][action] = reward + GAMMA * np.amax(t[0])
#             self.model.fit(state, target, epochs=1, verbose=0)
#         if self.epsilon > self.epsilon_min:
#             self.epsilon *= self.epsilon_decay

#     # 更新目标网络
#     def update_target_model(self):
#         self.target_model.set_weights(self.model.get_weights())

# # 构建强化学习环境
# class TradingEnvironment:
#     def __init__(self, data, initial_balance=10000):
#         self.data = data
#         self.index = 0
#         self.balance = initial_balance
#         self.position = 0
#         self.total_value = initial_balance

#     def reset(self):
#         self.index = 0
#         self.balance = 10000
#         self.position = 0
#         self.total_value = 10000
#         return self._get_state()

#     def _get_state(self):
#         return self.data.iloc[self.index, :].values.reshape(1, -1)  # 返回当前状态

#     def step(self, action):
#         # 动作解释：根据动作选择分配权重，调整投资组合
#         weight = action / (ACTION_SIZE - 1)  # 动作空间离散化：0%, 10%, ..., 100%
#         current_price = self.data['Clsprc'].iloc[self.index]
#         next_price = self.data['Clsprc'].iloc[self.index + 1]
#         reward = 0

#         # 模拟交易
#         self.position = self.total_value * weight
#         self.balance = self.total_value * (1 - weight)
#         new_value = self.position * (next_price / current_price) + self.balance
#         reward = new_value - self.total_value  # 奖励为资产变化
#         self.total_value = new_value

#         self.index += 1
#         done = self.index == len(self.data) - 1
#         next_state = self._get_state()
#         return next_state, reward, done

# # 初始化环境和智能体
# env = TradingEnvironment(data)
# agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

# # 训练过程
# EPISODES = 500
# for e in range(EPISODES):
#     state = env.reset()
#     for time in range(len(data) - 1):
#         action = agent.act(state)
#         next_state, reward, done = env.step(action)
#         agent.remember(state, action, reward, next_state, done)
#         state = next_state
#         if done:
#             print(f"Episode {e + 1}/{EPISODES} - Total Value: {env.total_value:.2f}")
#             break
#     agent.replay(BATCH_SIZE)
#     agent.update_target_model()


### 期末作业代码部分结束
