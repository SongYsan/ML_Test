import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


#数据预处理..
# x = pd.read_csv('data.csv')
# #ss = StandardScaler()
# ss = MinMaxScaler()
# scale_features = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
# x[scale_features] = ss.fit_transform(x[scale_features])
# x.to_csv("data_fixed.csv",index=0)#归一化之后保存为csv文件，然后再读取为numpy数组


#加载训练数据集 五个分类 每一类2000行数据
tmp = np.loadtxt("data.csv",dtype=np.str,delimiter=',')
# train_data = tmp[4001:14001,0:54].astype(np.float)#加载数据部分
# train_label = tmp[4001:14001,54].astype(np.float)#加载类别标签部分
train_data = tmp[1:8001,0:10].astype(np.float)#加载数据部分
train_label = tmp[1:8001,10].astype(np.float)#加载类别标签部分


#分别加载测试数据集 五个分类 每一类500行数据
test_data_class3 = tmp[8001:8501,0:10].astype(np.float)#加载数据部分
test_label_class3 = tmp[8001:8501,10].astype(np.float)#加载类别标签部分

test_data_class4 = tmp[8501:9001,0:10].astype(np.float)#加载数据部分
test_label_class4 = tmp[8501:9001,10].astype(np.float)#加载类别标签部分

test_data_class5 = tmp[9001:9501,0:10].astype(np.float)#加载数据部分
test_label_class5 = tmp[9001:9501,10].astype(np.float)#加载类别标签部分

test_data_class6 = tmp[9501:10001,0:10].astype(np.float)#加载数据部分
test_label_class6 = tmp[9501:10001,10].astype(np.float)#加载类别标签部分


test_list = [test_label_class3, test_label_class4, test_label_class5, test_label_class6]


# #decision trees
#clf = tree.DecisionTreeClassifier().fit(train_data,train_label)

#SVM
#clf = svm.SVC(gamma='scale',decision_function_shape='ovo').fit(train_data,train_label)

#GradientBoostingClassifier
#clf = GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1,random_state=0).fit(train_data,train_label)

#KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3).fit(train_data, train_label)

#用训练好的模型（上面三种，gnb、dt、ab 自己换）对测试集进行预测
prediction3 = clf.predict(test_data_class3)
prediction4 = clf.predict(test_data_class4)
prediction5 = clf.predict(test_data_class5)
prediction6 = clf.predict(test_data_class6)

pre = [prediction3, prediction4, prediction5, prediction6]

#统计预测准确率
for i in range(4):
    count = 0
    for j in range(len(test_label_class3)):
        if(pre[i][j] == test_list[i][j]):
            count += 1
    # print("Accuracy of classification " + str(i+3) +" : %.3f" % (count / 500))
    print("%.3f" % (count / 500))





