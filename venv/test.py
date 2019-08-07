# from sklearn import datasets
# rawData = datasets.load_files("data_folder")
# print(rawData.filenames)

# import numpy as np
# tmp = np.loadtxt("cover.csv",dtype=np.str,delimiter=',')
# data = tmp[1:,0:8].astype(np.float)#加载数据部分
# label = tmp[1:,9].astype(np.float)#加载类别标签部分
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
# x = pandas.read_csv('cover.csv')
# ss = StandardScaler()
# scale_feature = ['col1','col2','col3','col4','col5','col6','col7','col8','col9']
# x[scale_feature] = ss.fit_transform(x[scale_feature])
# x.to_csv("cover_fix.csv",index=0)#归一化之后保存为csv文件，然后再读取为numpy数组

tmp = np.loadtxt("cover.csv",dtype=np.str,delimiter=',')
data = tmp[1:,0:8].astype(np.float)#加载数据部分
label = tmp[1:,9].astype(np.float)#加载类别标签部分

# test_data = tmp[801:,0:8].astype(np.float)#加载数据部分
# test_label = tmp[801:,9].astype(np.float)#加载类别标签部分


#Gaussian Naive Bayes贝叶斯方法训练
gnb = GaussianNB()
x_pred = gnb.fit(data,label).predict(data)
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (data.shape[0],(label != x_pred).sum()))





