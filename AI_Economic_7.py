# -*- coding: utf-8 -*-

###输出警告改动：屏蔽警告 
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 改动：只显示错误信息 

######################################7.1数据
import os
import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn-v0_8')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
pd.options.display.precision = 4
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'


# 修改后代码
raw = pd.read_csv('TRD_Dalyr.csv')  # 加载新数据集
symbol = 'Moutai'

# 转换日期并设置为索引
raw['Trddt'] = pd.to_datetime(raw['Trddt'])  # 将日期列转换为 datetime 类型
raw.set_index('Trddt', inplace=True)  # 将日期列设置为索引

# 选择 'Clsprc' 列作为收盘价数据，并重命名
data = pd.DataFrame(raw['Clsprc'])  # 选择收盘价列
data.columns = [symbol]  # 重命名为 'Moutai'（茅台）

print(data.head())  # 输出处理后数据前5行信息
data.info()  # 输出新数据集基本信息
data.plot(figsize=(10, 6))  # 绘制每日收盘价趋势：绘图1




##########################7.2基线预测

###创建滞后特征
lags = 10  # 改动：滞后设置为10

def add_lags(data, symbol, lags, window=30):  # 改动：滚动窗口设置为30
    cols = []
    df = data.copy()
    df.dropna(inplace=True)
    df['r'] = np.log(df / df.shift())
    df['sma'] = df[symbol].rolling(window).mean()
    df['min'] = df[symbol].rolling(window).min()
    df['max'] = df[symbol].rolling(window).max()
    df['mom'] = df['r'].rolling(window).mean()
    df['vol'] = df['r'].rolling(window).std()
    df.dropna(inplace=True)
    df['d'] = np.where(df['r'] > 0, 1, 0)
    features = [symbol, 'r', 'd', 'sma', 'min', 'max', 'mom', 'vol']
    for f in features:
        for lag in range(1, lags + 1):
            col = f'{f}_lag_{lag}'
            df[col] = df[f].shift(lag)
            cols.append(col)
    df.dropna(inplace=True)
    return df, cols

data, cols = add_lags(data, symbol, lags)


###类别不平衡
len(data)
data.iloc[:10, :14].round(4)

#显示两个类的频率
c = data['d'].value_counts()
print(c)

#计算适当的权重以达到相等的权重
def cw(df):
    c0, c1 = np.bincount(df['d'])
    w0 = (1 / c0) * (len(df)) / 2
    w1 = (1 / c1) * (len(df)) / 2
    return {0: w0, 1: w1}

class_weight = cw(data)
class_weight

#根据计算出的权重，两个类的权重相等
class_weight[0] * c[0]

class_weight[1] * c[1]


###Keras创建DNN模型样本内
import random
import logging
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
tf.get_logger().setLevel(logging.ERROR)

def set_seeds(seed=100):
    random.seed(seed) #设定Python随机数种子
    np.random.seed(seed) #设定Numpy随机数种子
    tf.random.set_seed(seed) #设定TensorFlow随机数种子
    
optimizer = Adam(learning_rate=0.0015) #默认优化器    

    
def create_model_1(hl=1, hu=128, optimizer=optimizer):  
    optimizer = Adam(learning_rate=0.0015) #默认优化器
    model = Sequential()
    model.add(Dense(hu, input_dim=len(cols),
                    activation='relu')) #第1层
    for _ in range(hl):
        model.add(Dense(hu, activation='relu')) #中间层
    model.add(Dense(1, activation='sigmoid')) #输出层
    model.compile(loss='binary_crossentropy', #损失函数
                  optimizer=optimizer, #使用的优化器
                  metrics=['accuracy']) #要收集的其它指标
    return model    
    
set_seeds()
model = create_model_1(hl=1, hu=128)   

model.fit(data[cols], data['d'], epochs=50,
          verbose=False, class_weight=cw(data)) #考虑权重

model.evaluate(data[cols], data['d'])  # 输出语句1：在训练数据上评估模型性能

data['p'] = np.where(model.predict(data[cols]) > 0.5, 1, 0)
data['p'].value_counts()

##样本内基线性能约为50%



###Keras创建DNN模型样本外
split = int(len(data) * 0.8) #数据集拆分

train = data.iloc[:split].copy() #训练集

test = data.iloc[split:].copy() #测试集

set_seeds()
model = create_model_1(hl=1, hu=128)

h = model.fit(train[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False,
          class_weight=cw(train))

model.evaluate(train[cols], train['d'])  # 输出语句2：样本内评估性能

model.evaluate(test[cols], test['d'])  # 输出语句3：样本外（测试集）评估性能

test['p'] = np.where(model.predict(test[cols]) > 0.5, 1, 0)

test['p'].value_counts()

res = pd.DataFrame(h.history)

res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')  # 绘图2
plt.ylim(0, 1)  # 设置纵轴范围为0到1
plt.show()


############################7.3归一化
mu, std = train.mean(), train.std() #所有训练特征的均值和标准差
train_ = (train - mu) / std #训练数据集归一化
train_.std().round(3)
   
set_seeds()
model = create_model_1(hl=2, hu=128)

h = model.fit(train_[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False,
          class_weight=cw(train))

model.evaluate(train_[cols], train['d'])  # 输出语句4：评估样本内性能，归一化后大幅提升

test_ = (test - mu) / std #测试集归一化

model.evaluate(test_[cols], test['d'])  # 输出语句5：评估样本外性能，归一化后提升不大

test['p'] = np.where(model.predict(test_[cols]) > 0.5, 1, 0)
test['p'].value_counts()

res = pd.DataFrame(h.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')  # 绘图3
plt.ylim(0, 1)  # 设置纵轴范围为0到1
plt.show()

###需要处理过拟合：暂退、正则化、装袋
###过拟合表现为：样本内训练集测试效果'accuracy'高, 但样本外测试集测试效果'val_accuracy'差
###处理过拟合：结果应当使两者接近



################################7.4暂退

from tensorflow.keras.layers import Dropout

def create_model_2(hl=1, hu=128, dropout=True, rate=0.4,  # 改动：可能调整暂退率
                 optimizer=optimizer):
    model = Sequential()
    model.add(Dense(hu, input_dim=len(cols),
                    activation='relu'))
    if dropout:
        model.add(Dropout(rate, seed=100)) #在每一层后加入暂退
    for _ in range(hl):
        model.add(Dense(hu, activation='relu'))
        if dropout:
            model.add(Dropout(rate, seed=100)) #在每一层后加入暂退
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                 metrics=['accuracy'])
    return model

set_seeds()
model = create_model_2(hl=1, hu=128, rate=0.4)

h = model.fit(train_[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.15, shuffle=False,
          class_weight=cw(train))

model.evaluate(train_[cols], train['d'])  # 输出语句6：样本内，仅暂退后

model.evaluate(test_[cols], test['d'])  # 输出语句7：样本外，仅暂退后

res = pd.DataFrame(h.history)

res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')  # 绘图4
plt.ylim(0, 1)  # 设置纵轴范围为0到1
plt.show()

###训练集准确率和验证集准确率没有像以前那样迅速分开



########################7.5正则化
from tensorflow.keras.regularizers import l1, l2

def create_model_3(hl=1, hu=128, dropout=False, rate=0.4, 
                 regularize=False, reg=l1(0.001),#正则化添加到每一层
                 optimizer=optimizer, input_dim=len(cols)):
    if not regularize:
        reg = None
    model = Sequential()
    model.add(Dense(hu, input_dim=input_dim,
                    activity_regularizer=reg,
                    activation='relu'))
    if dropout:
        model.add(Dropout(rate, seed=100))
    for _ in range(hl):
        model.add(Dense(hu, activation='relu',
                        activity_regularizer=reg)) #正则化添加到每一层
        if dropout:
            model.add(Dropout(rate, seed=100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                 metrics=['accuracy'])
    return model

set_seeds()
model = create_model_3(hl=1, hu=128, regularize=True)

# 在每次调用之前创建新优化器实例:改动
optimizer_instance = Adam(learning_rate=0.0015)  # 动态实例化优化器
model = create_model_3(hl=1, hu=128, regularize=True, optimizer=optimizer_instance)

h = model.fit(train_[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False,
          class_weight=cw(train))

model.evaluate(train_[cols], train['d'])  # 输出语句8：样本内，仅正则化后

model.evaluate(test_[cols], test['d'])  # 输出语句9,：样本外，仅正则化后

res = pd.DataFrame(h.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')  # 绘图5
plt.ylim(0, 1)  # 设置纵轴范围为0到1
plt.show()


###暂退和正则化一起使用
set_seeds()
h = model = create_model_3(hl=2, hu=128,
                     dropout=True, rate=0.4,
                     regularize=True, reg=l2(0.0001),
                    )

# 在每次调用之前创建新优化器实例:改动 
optimizer_instance = Adam(learning_rate=0.0015)  # 动态实例化优化器
model = create_model_3(hl=1, hu=128, regularize=True, optimizer=optimizer_instance)

h = model.fit(train_[cols], train['d'],
          epochs=50, verbose=False,
          validation_split=0.2, shuffle=False,
          class_weight=cw(train))

model.evaluate(train_[cols], train['d'])  # 输出语句10：样本内，暂退和正则化一起用后

model.evaluate(test_[cols], test['d'])  # 输出语句11：样本外，暂退和正则化一起用后

res = pd.DataFrame(h.history)
res[['accuracy', 'val_accuracy']].plot(figsize=(10, 6), style='--')  # 绘图6
plt.ylim(0, 1)  # 设置纵轴范围为0到1
plt.show()

res.mean()['accuracy'] - res.mean()['val_accuracy']

###过拟合进一步缓和


#########################7.6装袋
from sklearn.ensemble import BaggingClassifier
from scikeras.wrappers import KerasClassifier

len(cols)

max_features = 0.75

set_seeds()
base_estimator = KerasClassifier(
        # 改动：增加一大段 
        build_fn=lambda **kwargs: create_model_3(
        hl=kwargs.get('hl', 1),
        hu=kwargs.get('hu', 128),
        dropout=kwargs.get('dropout', True),
        rate=kwargs.get('rate', 0.3),
        regularize=kwargs.get('regularize', False),
        reg=kwargs.get('reg', l1(0.0005)),
        optimizer=Adam(learning_rate=0.0015),  # 每次创建新的优化器实例
        input_dim=kwargs.get('input_dim', len(cols))
    ),
                        verbose=False, epochs=20, hl=1, hu=128,
                        dropout=True, regularize=False,
                        #基础估计器，这里是keras的Sequential模型的实例
                        input_dim=int(len(cols) * max_features))

model_bag = BaggingClassifier(estimator=base_estimator, 
                          n_estimators=15,
                          max_samples=0.75,
                          max_features=max_features,
                          bootstrap=True,
                          bootstrap_features=True,
                          n_jobs=1,
                          random_state=100,
                         ) #BaggingClassifier模型被实例化为许多相同的基础估计器

model_bag.fit(train_[cols], train['d'])

train_accuracy = model_bag.score(train_[cols], train['d'])
print(f"In-sample-Accuracy: {train_accuracy:.4f}")  # 装袋后，样本内。改动：加输出 

test_accuracy = model_bag.score(test_[cols], test['d'])
print(f"Out-sample-Accuracy: {test_accuracy:.4f}")  # 装袋后，样本外。改动：加输出 

test['p'] = model_bag.predict(test_[cols])

print("Prediction Value Counts (Test):")
print(test['p'].value_counts())  # 改动：加输出 


###结果可能有类不平衡驱动


##################################7.7优化器

import time
optimizers = ['sgd', 'rmsprop', 'adagrad', 'adadelta',
              'adam', 'adamax', 'nadam']

for optimizer in optimizers:
    set_seeds()
    model = create_model_3(hl=1, hu=128,
                     dropout=True, rate=0.4,
                     regularize=False, reg=l2(0.01),
                     optimizer=optimizer
                    ) #为给定的优化器实例化DNN模型
    t0 = time.time()
    model.fit(train_[cols], train['d'],
              epochs=50, verbose=False,
              validation_split=0.2, shuffle=False,
              class_weight=cw(train)) #使用给定的优化器拟合模型
    t1 = time.time()
    t = t1 - t0
    #评估样本内性能
    acc_tr = model.evaluate(train_[cols], train['d'], verbose=False)[1]
    #评估样本外性能
    acc_te = model.evaluate(test_[cols], test['d'], verbose=False)[1]
    out = f'{optimizer:10s} | time[s]: {t:.4f} | in-sample={acc_tr:.4f}'
    out += f' | out-of-sample={acc_te:.4f}'
    print(out)

