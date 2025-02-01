
#输出utf8格式  
import sys
sys.stdout.reconfigure(encoding='utf-8')

##################6.1有效市场

####EMH半正式测试：取一段金融时间序列，多次滞后价格数据
####，并以滞后的价格数据为特征，以当前价格为标签，输入OLS回归模型

import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision=4
np.set_printoptions(suppress=True, precision=4)

###数据准备
data= pd.read_csv('aiif_eikon_eod_data_4.csv', index_col=0, parse_dates=True).dropna()

#绘制归一化时间序列数据
(data / data.iloc[0]).plot(figsize=(10, 6), cmap='coolwarm')

#定义参数滞后时间（对于交易日而言）
lags = 7

def add_lags(data, ric, lags):
    cols = []
    df = pd.DataFrame(data[ric])
    for lag in range(1, lags + 1):
        col = 'lag_{}'.format(lag) #创建列名
        df[col] = df[ric].shift(lag) #滞后价格序列
        cols.append(col) #以列表存储列名
    df.dropna(inplace=True) #删除不完整的数据行
    return df, cols


dfs = {}
for sym in data.columns:
    df, cols = add_lags(data, sym, lags) #滞后的每一个金融时间序列
    dfs[sym] = df #以字典存储结果

dfs[sym].head(7) #展示一个滞后价格数据的样例

###OLS回归分析
regs = {}
for sym in data.columns:
    df = dfs[sym] #获取当前时间序列数据
    reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0] #回归分析
    regs[sym] = reg

#拼接多时间序列的最优结果，以单个数组对象存储
rega = np.stack(tuple(regs.values()))

#结果存入数据框并进行展示
regd = pd.DataFrame(rega, columns=cols, index=data.columns)
regd

#可视化每一个滞后时间的多个最优参数（权重）结果的平均值
regd.mean().plot(kind='bar', figsize=(10, 6))

####弱式EMH的强力证据
####回归分析违反了几个假设：独立、平稳性。
####增加相关性，对结果改进不大

#展示滞后时间序列的相关性
dfs[sym].corr()

#使用迪基-福勒检验（Dickey-Fuller test）平稳性检验
from statsmodels.tsa.stattools import adfuller
adfuller(data[sym].dropna())


#####################6.2基于收益数据的市场预测
rets = np.log(data / data.shift(1))
rets.dropna(inplace=True)

dfs = {}
for sym in data:
    df, cols = add_lags(rets, sym, lags) #滞后对数收益率数据
    mu, std = df[cols].mean(), df[cols].std() 
    df[cols] = (df[cols] - mu) / std  #对特征数据进行高斯归一化
    dfs[sym] = df

dfs[sym].head()

#平稳性检验
adfuller(dfs[sym]['lag_1'])

#数据相关性
dfs[sym].corr()


###OLS检验
from sklearn.metrics import accuracy_score

for sym in data:
    df = dfs[sym]
    reg = np.linalg.lstsq(df[cols], df[sym], rcond=-1)[0] #OLS回归
    pred = np.dot(df[cols], reg) #模型预测
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred)) #准确率分析
    print(f'OLS | {sym:10s} | acc={acc:.4f}')

##OLS回归的方法预测明日市场走向的准确率略高于50%


######基于scikit-learn包的神经网络模型
from sklearn.neural_network import MLPRegressor

for sym in data.columns:
    df = dfs[sym]
    model = MLPRegressor(hidden_layer_sizes=[512],
                         random_state=100,
                         max_iter=1000,
                         early_stopping=True,
                         validation_fraction=0.15,
                         shuffle=False) #模型实例化
    model.fit(df[cols], df[sym]) #模型拟合
    pred = model.predict(df[cols]) #模型预测
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred)) #准确率计算
    print(f'MLP | {sym:10s} | acc={acc:.4f}')

########keras包构建神经网络模型
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

np.random.seed(100)
tf.random.set_seed(100)

def create_model(problem='regression'): #模型创建函数
    model = Sequential()
    model.add(Dense(512, input_dim=len(cols),
                    activation='relu'))
    if problem == 'regression':
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
    else:
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

for sym in data.columns[:]:
    df = dfs[sym]
    model = create_model() #模型实例化
    model.fit(df[cols], df[sym], epochs=25, verbose=False) #模型拟合
    pred = model.predict(df[cols]) #模型预测
    acc = accuracy_score(np.sign(df[sym]), np.sign(pred)) #模型准确率分析
    print(f'DNN | {sym:10s} | acc={acc:.4f}')


##################样本外的表现
#80%训练，20%测试
split = int(len(dfs[sym]) * 0.8)

###OLS回归
for sym in data.columns:
    df = dfs[sym]
    train = df.iloc[:split] #创建训练集
    reg = np.linalg.lstsq(train[cols], train[sym], rcond=-1)[0]
    test = df.iloc[split:] #创建测试集
    pred = np.dot(test[cols], reg)
    acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
    print(f'OLS | {sym:10s} | acc={acc:.4f}')

##样本外与样本内近似


######基于scikit-learn包的神经网络模型
for sym in data.columns:
    df = dfs[sym]
    train = df.iloc[:split]
    model = MLPRegressor(hidden_layer_sizes=[512],
                         random_state=100,
                         max_iter=1000,
                         early_stopping=True,
                         validation_fraction=0.15,
                         shuffle=False)
    model.fit(train[cols], train[sym])
    test = df.iloc[split:]
    pred = model.predict(test[cols])
    acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
    print(f'MLP | {sym:10s} | acc={acc:.4f}')

###样本外的表现比样本内差得多，结果接近于OLS回归


########keras包构建神经网络模型
for sym in data.columns:
    df = dfs[sym]
    train = df.iloc[:split]
    model = create_model()
    model.fit(train[cols], train[sym], epochs=50, verbose=False)
    test = df.iloc[split:]
    pred = model.predict(test[cols])
    acc = accuracy_score(np.sign(test[sym]), np.sign(pred))
    print(f'DNN | {sym:10s} | acc={acc:.4f}')

###准确率都在50%上下波动




########################6.3基于更多特征的市场预测
data= pd.read_csv('aiif_eikon_eod_data_4.csv', index_col=0, parse_dates=True).dropna()

def add_lags(data, ric, lags, window=50):
    cols = []
    df = pd.DataFrame(data[ric])
    df.dropna(inplace=True)
    df['r'] = np.log(df / df.shift())
    df['sma'] = df[ric].rolling(window).mean() #简单移动平均
    df['min'] = df[ric].rolling(window).min() #滚动最小值
    df['max'] = df[ric].rolling(window).max()#滚动最大值
    df['mom'] = df['r'].rolling(window).mean()#动量：对数收益滑动均值
    df['vol'] = df['r'].rolling(window).std()#滚动波动率
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)#当日收益率方向（二进制0、1特征）
    features = [ric, 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols


lags = 5
dfs = {}
for ric in data:
    df, cols = add_lags(data, ric, lags)
    dfs[ric] = df.dropna(), cols

len(cols)


######基于scikit-learn包的神经网络模型
from sklearn.neural_network import MLPClassifier

for ric in data:
    model = MLPClassifier(hidden_layer_sizes=[512],
                          random_state=100,
                          max_iter=1000,
                          early_stopping=True,
                          validation_fraction=0.15,
                          shuffle=False)
    df, cols = dfs[ric]
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    model.fit(df[cols], df['d'])
    pred = model.predict(df[cols])
    acc = accuracy_score(df['d'], pred)
    print(f'IN-SAMPLE | {ric:7s} | acc={acc:.4f}')

###MLPClassifier模型在样本内效果大大提升

########keras包构建神经网络模型
for ric in data:
    model = create_model('classification')
    df, cols = dfs[ric]
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    model.fit(df[cols], df['d'], epochs=50, verbose=False)
    pred = np.where(model.predict(df[cols]) > 0.5, 1, 0)
    acc = accuracy_score(df['d'], pred)
    print(f'IN-SAMPLE | {ric:7s} | acc={acc:.4f}')

###基于keras构建的Sequential模型在样本内效果能够达到70%


######基于scikit-learn包的神经网络模型，样本外，检验过拟合

def train_test_model(model):
    for ric in data:
        df, cols = dfs[ric]
        split = int(len(df) * 0.85)
        train = df.iloc[:split].copy()
        mu, std = train[cols].mean(), train[cols].std()
        train[cols] = (train[cols] - mu) / std
        model.fit(train[cols], train['d'])
        test = df.iloc[split:].copy() 
        test[cols] = (test[cols] - mu) / std
        pred = model.predict(test[cols])
        acc = accuracy_score(test['d'], pred)
        print(f'OUT-OF-SAMPLE | {ric:7s} | acc={acc:.4f}')
        
model_mlp = MLPClassifier(hidden_layer_sizes=[512],
                          random_state=100,
                          max_iter=1000,
                          early_stopping=True,
                          validation_fraction=0.15,
                          shuffle=False)

train_test_model(model_mlp)

##样本外表现显著差与样本内，存在过拟合

####装袋来处理过拟合

from sklearn.ensemble import BaggingClassifier

base_estimator = MLPClassifier(hidden_layer_sizes=[256],
                          random_state=100,
                          max_iter=1000,
                          early_stopping=True,
                          validation_fraction=0.15,
                          shuffle=False) #基础模型

model_bag = BaggingClassifier(estimator=base_estimator,
                          n_estimators=35, #使用模型量
                          max_samples=0.25, #每个模型使用的最大训练数据量
                          max_features=0.5, #每个模型使用的最大特征数
                          bootstrap=False, #是否可以重用数据
                          bootstrap_features=True, #是否可以重用特征
                          n_jobs=8, #并行线程数
                          random_state=100
                         )

train_test_model(model_bag)

###过拟合有所缓解



##########################6.4日内市场预测

data= pd.read_csv('aiif_eikon_id_data_6.csv', index_col=0, parse_dates=True)

data.tail()

data.info()

lags = 5

dfs = {}
for ric in data:
    df, cols = add_lags(data, ric, lags)
    dfs[ric] = df, cols

###MLPClassifier样本外
train_test_model(model_mlp)

###装袋
train_test_model(model_bag)

