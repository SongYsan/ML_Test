import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree.export import export_text
from sklearn.ensemble import AdaBoostClassifier


#数据预处理..
x = pd.read_csv('data.csv')
#ss = StandardScaler()
ss = MinMaxScaler()
scale_features = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
x[scale_features] = ss.fit_transform(x[scale_features])
x.to_csv("data_fixed.csv",index=0)#归一化之后保存为csv文件，然后再读取为numpy数组


#加载训练数据集 七个分类 每一类2000行数据
tmp = np.loadtxt("data_fixed.csv",dtype=np.str,delimiter=',')
train_data = tmp[1:14001,0:54].astype(np.float)#加载数据部分
train_label = tmp[1:14001,54].astype(np.float)#加载类别标签部分


#分别加载测试数据集 七个分类 每一类500行数据
test_data_class1 = tmp[14001:14501,0:54].astype(np.float)#加载数据部分
test_label_class1 = tmp[14001:14501,54].astype(np.float)#加载类别标签部分

test_data_class2 = tmp[14501:15001,0:54].astype(np.float)#加载数据部分
test_label_class2 = tmp[14501:15001,54].astype(np.float)#加载类别标签部分

test_data_class3 = tmp[15001:15501,0:54].astype(np.float)#加载数据部分
test_label_class3 = tmp[15001:15501,54].astype(np.float)#加载类别标签部分

test_data_class4 = tmp[15501:16001,0:54].astype(np.float)#加载数据部分
test_label_class4 = tmp[15501:16001,54].astype(np.float)#加载类别标签部分

test_data_class5 = tmp[16001:16501,0:54].astype(np.float)#加载数据部分
test_label_class5 = tmp[16001:16501,54].astype(np.float)#加载类别标签部分

test_data_class6 = tmp[16501:17001,0:54].astype(np.float)#加载数据部分
test_label_class6 = tmp[16501:17001,54].astype(np.float)#加载类别标签部分

test_data_class7 = tmp[17001:17501,0:54].astype(np.float)#加载数据部分
test_label_class7 = tmp[17001:17501,54].astype(np.float)#加载类别标签部分

test_list = [test_label_class1, test_label_class2, test_label_class3, test_label_class4, test_label_class5, test_label_class6, test_label_class7]


#Gaussian Naive Bayes贝叶斯方法训练
gnb = GaussianNB()
gnb = gnb.fit(train_data,train_label)

#decision trees
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_data,train_label)

#Adaboost
ab = AdaBoostClassifier(n_estimators=50)
ab = ab.fit(train_data,train_label)

#用训练好的模型（上面三种，gnb、dt、ab 自己换）对测试集进行预测
prediction1 = ab.predict(test_data_class1)
prediction2 = ab.predict(test_data_class2)
prediction3 = ab.predict(test_data_class3)
prediction4 = ab.predict(test_data_class4)
prediction5 = ab.predict(test_data_class5)
prediction6 = ab.predict(test_data_class6)
prediction7 = ab.predict(test_data_class7)

pre = [prediction1, prediction2, prediction3, prediction4, prediction5, prediction6, prediction7]
print(prediction2)

#统计预测准确率
for i in range(7):
    count = 0
    for j in range(len(test_label_class1)):
        if(pre[i][j] == test_list[i][j]):
            count += 1
    print("Accuracy of classification " + str(i+1) +" : %.2f" % (count / 500))






