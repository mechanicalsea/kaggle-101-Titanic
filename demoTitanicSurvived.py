"""
=======================================
Titanic: Machine Learning from Disaster
=======================================

Accuracy: 0.77990

0-1 分类：根据
    乘客类型，姓名，性别，年龄，兄弟个数，父子个数，船票，票价，船舱，港口
    判定乘客是船难中存活下来

数据分析与处理：
  1. 需要对 港口，年龄，船票 项做缺失项补全
  2. 对 性别，港口 项做字典替换
  3. 除去无用信息
  4. 利用 Pandas 工具包处理数据

分类：
  1. 可通过组合加入新的特征，即人工特征
  2. 可以对特征做标准化处理
  3. 多种分类方法同时判定，然后加权获得判定结果

总结：
  1. 增加特征有利于分类
  2. 随机森林在分类上确实很好用
  3. 组合分类方法，有点作用
  
"""
print(__doc__)

import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation  #导入交叉检验

# 数据缺失项补齐的设定
#     df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
pd.set_option('chained_assignment',None)

#######################################################################################
# 数据集读取，return 数据集，编号，港口字典
def csvFileRead(filename, Ports_dict):

    # pandas 的数据读取
    df = pd.read_csv(filename, header=0)
    
    # Sex 中 'female':0, 'male':1 替换，并生成新的一列 Gender
    df['Gender'] = df['Sex'].map({'female':0, 'male':1}).astype(int)

    # 对 Embarked 项做补缺失项操作
    if len(df.Embarked[df.Embarked.isnull()]) > 0:
        df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
        
    # 建立 Embarked 项字典，并映射为 ID, 'S','Q','C'
    if not Ports_dict:
        Ports = list(enumerate(np.unique(df['Embarked'])))
        Ports_dict = {name : i-1 for i, name in Ports}
    df.Embarked = df.Embarked.map(lambda x: Ports_dict[x]).astype(int)

    # Age 缺失项补齐为平均年龄
    median_age = df['Age'].dropna().median()
    if len(df.Age[df.Age.isnull()]) > 0:
        df.loc[(df.Age.isnull()), 'Age'] = median_age

    # Fare 缺失项补齐为各类船票的均值
    if len(df.Fare[df.Fare.isnull()]) > 0:
        median_fare = np.zeros(3)
        for f in range(0,3):
            median_fare[f] = df[df.Pclass == f+1]['Fare'].dropna().median()
        for f in range(0,3):
            df.loc[(df.Fare.isnull()) & (df.Pclass == f+1), 'Fare'] = median_fare[f]

    # 提取 PassengerId 项
    ids = df['PassengerId'].values

    # 舍去无用特征
    df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

    return df, ids, Ports_dict
#######################################################################################
# 结果存储
def csvSave(filename, ids, predicted):
    with open('result.csv', 'w') as mycsv:
        mywriter = csv.writer(mycsv)
        mywriter.writerow(["PassengerId","Survived"])
        mywriter.writerows(zip(ids, predicted))

#######################################################################################
# 测试对比
def resultCompare(predicted):
    file1 = 'gendermodel.csv'
    file2 = 'gendermodel.csv'
    df1 = pd.read_csv(file1, header=0)
    df2 = pd.read_csv(file2, header=0)
    result1 = df1.Survived.values
    result2 = df2.Survived.values
    accuracy1 = 1.0 - sum(abs(result1-predicted))/len(predicted)
    accuracy2 = 1.0 - sum(abs(result2-predicted))/len(predicted)
    print('[1]: ', accuracy1, '\n[2]: ', accuracy2)

#######################################################################################
# 分类
def classificationResult(train_df, test_df):
    train_X = train_df.values[:,1:]
    train_Y = train_df.values[:,0]
    test_X = test_df.values

    print('添加人工特征...')
    temp1 = train_X[:,2]*train_X[:,3]
    temp2 = train_X[:,4]/train_X[:,1]
    temp3 = train_X[:,5]*train_X[:,4]
    temp4 = train_X[:,0]*train_X[:,4]
    train_X = np.c_[train_X, temp1, temp2, temp3, temp4]

    temp1 = test_X[:,2]*test_X[:,3]
    temp2 = test_X[:,4]/test_X[:,1]
    temp3 = test_X[:,5]*test_X[:,4]
    temp4 = test_X[:,0]*test_X[:,4]
    test_X = np.c_[test_X, temp1, temp2, temp3, temp4]
    
    print('特征标准化...')
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)


    print('训练中...')
    classifier = RandomForestClassifier(n_estimators=100)
    clf1 = LogisticRegression(C=10000)
    clf2 = AdaBoostClassifier()
    #scores=cross_validation.cross_val_score(classifier,train_X,train_Y,cv=5)  #交叉检验
    #print(scores,scores.mean())

    classifier.fit(train_X, train_Y)
    clf1.fit(train_X, train_Y)
    clf2.fit(train_X, train_Y)
    

    print('预测中...')
    predicted = 0.6*classifier.predict(test_X).astype(int)
    predicted += 0.2*clf1.predict(test_X).astype(int)
    predicted += 0.2*clf2.predict(test_X).astype(int)
    
    return (predicted >= 0.75).astype(int)

#######################################################################################
# main
train_df, train_ids, Ports_dict = csvFileRead('train.csv', {})
test_df, ids, Ports_dict = csvFileRead('test.csv', Ports_dict)
predicted = classificationResult(train_df, test_df)
csvSave('result.csv', ids, predicted)

print('完成.')

resultCompare(predicted)
