# 输出 utf8 格式
import sys
sys.stdout.reconfigure(encoding='utf-8')

################## 6.1 有效市场

#### EMH 半正式测试：取一段金融时间序列，多次滞后价格数据
#### 并以滞后的价格数据为特征，以当前价格为标签，输入 OLS 回归模型

import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision = 4
np.set_printoptions(suppress=True, precision=4)

### 数据准备
# 读取新的数据集 TRD_Dalyr.csv，仅保留日期和收盘价列，并将日期作为索引
data = pd.read_csv('TRD_Dalyr.csv', usecols=['Trddt', 'Clsprc'], parse_dates=['Trddt'], index_col='Trddt')
data.rename(columns={'Clsprc': 'Close'}, inplace=True)  # 将收盘价列重命名为 'Close'

# 绘制归一化时间序列数据
(data / data.iloc[0]).plot(figsize=(10, 6), cmap='coolwarm')

# 定义参数滞后时间（对于交易日而言）
lags = 7

# 滞后特征生成函数
def add_lags(data, lags):
    cols = []
    df = pd.DataFrame(data['Close'])
    for lag in range(1, lags + 1):
        col = f'lag_{lag}'  # 创建列名
        df[col] = df['Close'].shift(lag)  # 滞后价格序列
        cols.append(col)  # 以列表存储列名
    df.dropna(inplace=True)  # 删除不完整的数据行
    return df, cols

# 生成滞后特征数据
df, cols = add_lags(data, lags)
dfs = {'Close': df}  # 使用字典存储滞后特征数据

dfs['Close'].head(7)  # 展示滞后价格数据的样例

### OLS 回归分析
regs = {}
df = dfs['Close']  # 获取当前时间序列数据
reg = np.linalg.lstsq(df[cols], df['Close'], rcond=-1)[0]  # 回归分析
regs['Close'] = reg

# 拼接多时间序列的最优结果，以单个数组对象存储
rega = np.stack(tuple(regs.values()))

# 结果存入数据框并进行展示
regd = pd.DataFrame(rega, columns=cols, index=['Close'])
regd

# 可视化每一个滞后时间的多个最优参数（权重）结果的平均值
regd.mean().plot(kind='bar', figsize=(10, 6))

#### 弱式 EMH 的强力证据
#### 回归分析违反了几个假设：独立、平稳性。
#### 增加相关性，对结果改进不大

# 展示滞后时间序列的相关性
dfs['Close'].corr()

# 使用迪基-福勒检验（Dickey-Fuller test）平稳性检验
from statsmodels.tsa.stattools import adfuller
adfuller(data['Close'].dropna())


##################### 6.2 基于收益数据的市场预测
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)

# 生成滞后对数收益率数据
dfs = {}
df, cols = add_lags(rets, lags)
mu, std = df[cols].mean(), df[cols].std()
df[cols] = (df[cols] - mu) / std  # 对特征数据进行高斯归一化
dfs['Close'] = df

dfs['Close'].head()

# 平稳性检验
adfuller(dfs['Close']['lag_1'])

# 数据相关性
dfs['Close'].corr()


### OLS 检验
from sklearn.metrics import accuracy_score

df = dfs['Close']
reg = np.linalg.lstsq(df[cols], df['Close'], rcond=-1)[0]  # OLS 回归
pred = np.dot(df[cols], reg)  # 模型预测
acc = accuracy_score(np.sign(df['Close']), np.sign(pred))  # 准确率分析
print(f'OLS | Close | OLS回归预测明日市场走向的准确率 | acc={acc:.4f}')

## OLS 回归的方法预测明日市场走向的准确率略高于 50%


###### 基于 scikit-learn 包的神经网络模型
from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=[512],
                     random_state=100,
                     max_iter=1000,
                     early_stopping=True,
                     validation_fraction=0.15,
                     shuffle=False)  # 模型实例化
model.fit(df[cols], df['Close'])  # 模型拟合
pred = model.predict(df[cols])  # 模型预测
acc = accuracy_score(np.sign(df['Close']), np.sign(pred))  # 准确率计算
print(f'MLP | Close | 基于scikit-learn包的神经网络模型预测准确率 | acc={acc:.4f}')

######## keras 包构建神经网络模型
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

np.random.seed(100)
tf.random.set_seed(100)

def create_model(problem='regression'):  # 模型创建函数
    model = Sequential()
    model.add(Dense(512, input_dim=len(cols), activation='relu'))
    if problem == 'regression':
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

model = create_model()  # 模型实例化
model.fit(df[cols], df['Close'], epochs=25, verbose=False)  # 模型拟合
pred = model.predict(df[cols])  # 模型预测
acc = accuracy_score(np.sign(df['Close']), np.sign(pred))  # 模型准确率分析
print(f'DNN | Close | keras包构建神经网络模型预测准确率 | acc={acc:.4f}')


################## 样本外的表现
# 80% 训练，20% 测试
split = int(len(df) * 0.8)

### OLS 回归
train = df.iloc[:split]  # 创建训练集
reg = np.linalg.lstsq(train[cols], train['Close'], rcond=-1)[0]
test = df.iloc[split:]  # 创建测试集
pred = np.dot(test[cols], reg)
acc = accuracy_score(np.sign(test['Close']), np.sign(pred))
print(f'OLS | Close | OLS方法在样本外测试准确率 | acc={acc:.4f}')

## 样本外与样本内近似


###### 基于 scikit-learn 包的神经网络模型
train = df.iloc[:split]
model = MLPRegressor(hidden_layer_sizes=[512],
                     random_state=100,
                     max_iter=1000,
                     early_stopping=True,
                     validation_fraction=0.15,
                     shuffle=False)
model.fit(train[cols], train['Close'])
test = df.iloc[split:]
pred = model.predict(test[cols])
acc = accuracy_score(np.sign(test['Close']), np.sign(pred))
print(f'MLP | Close | scikit-learn方法在样本外测试准确率 | acc={acc:.4f}')

### 样本外的表现比样本内差得多，结果接近于 OLS 回归


######## keras 包构建神经网络模型
train = df.iloc[:split]
model = create_model()
model.fit(train[cols], train['Close'], epochs=50, verbose=False)
test = df.iloc[split:]
pred = model.predict(test[cols])
acc = accuracy_score(np.sign(test['Close']), np.sign(pred))
print(f'DNN | Close | keras方法在样本外测试准确率 | acc={acc:.4f}')

### 准确率都在 50% 上下波动




######################## 6.3 基于更多特征的市场预测

def add_lags(data, lags, window=50):
    cols = []
    df = pd.DataFrame(data['Close'])
    df.dropna(inplace=True)
    df['r'] = np.log(df / df.shift())
    df['sma'] = df['Close'].rolling(window).mean()  # 简单移动平均
    df['min'] = df['Close'].rolling(window).min()  # 滚动最小值
    df['max'] = df['Close'].rolling(window).max()  # 滚动最大值
    df['mom'] = df['r'].rolling(window).mean()  # 动量：对数收益滑动均值
    df['vol'] = df['r'].rolling(window).std()  # 滚动波动率
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)  # 当日收益率方向（二进制0、1特征）
    # 添加滞后特征
    features = ['Close', 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols

# 生成多特征滞后数据
lags = 5
df, cols = add_lags(data, lags)
dfs = {'Close': df}

len(cols)


###### 基于 scikit-learn 包的神经网络模型
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=[512],
                      random_state=100,
                      max_iter=1000,
                      early_stopping=True,
                      validation_fraction=0.15,
                      shuffle=False)
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()  # 标准化特征
model.fit(df[cols], df['d'])
pred = model.predict(df[cols])
acc = accuracy_score(df['d'], pred)
print(f'基于更多特征的市场预测 | MLP | IN-SAMPLE | Close | acc={acc:.4f}')

### MLPClassifier 模型在样本内效果大大提升

######## keras 包构建神经网络模型
model = create_model('classification')
df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()  # 标准化特征
model.fit(df[cols], df['d'], epochs=50, verbose=False)
pred = np.where(model.predict(df[cols]) > 0.5, 1, 0)
acc = accuracy_score(df['d'], pred)
print(f'基于更多特征的市场预测 | DNN | IN-SAMPLE | Close | acc={acc:.4f}')

### 基于 keras 构建的 Sequential 模型在样本内效果能够达到 70%


###### 基于 scikit-learn 包的神经网络模型，样本外，检验过拟合

def train_test_model(model):
    split = int(len(df) * 0.85)
    train = df.iloc[:split].copy()
    mu, std = train[cols].mean(), train[cols].std()
    train[cols] = (train[cols] - mu) / std
    model.fit(train[cols], train['d'])
    test = df.iloc[split:].copy()
    test[cols] = (test[cols] - mu) / std
    pred = model.predict(test[cols])
    acc = accuracy_score(test['d'], pred)
    print(f'检验过拟合 | MLP | OUT-OF-SAMPLE | Close | acc={acc:.4f}')

model_mlp = MLPClassifier(hidden_layer_sizes=[512],
                          random_state=100,
                          max_iter=1000,
                          early_stopping=True,
                          validation_fraction=0.15,
                          shuffle=False)

train_test_model(model_mlp)

## 样本外表现显著差于样本内，存在过拟合

#### 装袋来处理过拟合

from sklearn.ensemble import BaggingClassifier

base_estimator = MLPClassifier(hidden_layer_sizes=[256],
                               random_state=100,
                               max_iter=1000,
                               early_stopping=True,
                               validation_fraction=0.15,
                               shuffle=False)  # 基础模型

model_bag = BaggingClassifier(estimator=base_estimator,
                              n_estimators=35,  # 使用模型量
                              max_samples=0.25,  # 每个模型使用的最大训练数据量
                              max_features=0.5,  # 每个模型使用的最大特征数
                              bootstrap=False,  # 是否可以重用数据
                              bootstrap_features=True,  # 是否可以重用特征
                              n_jobs=8,  # 并行线程数
                              random_state=100
                             )

train_test_model(model_bag)

### 过拟合有所缓解



########################## 6.4 日内市场预测

# 假设有日内数据集 aiif_eikon_id_data_6.csv，直接读取并进行处理
# data = pd.read_csv('aiif_eikon_id_data_6.csv', index_col=0, parse_dates=True)

# data.tail()

# data.info()

# lags = 5

# # 为日内数据生成滞后特征
# dfs = {}
# df, cols = add_lags(data, lags)
# dfs['Close'] = df

# ### MLPClassifier 样本外
# train_test_model(model_mlp)

# ### 装袋
# train_test_model(model_bag)
