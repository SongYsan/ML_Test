import pandas
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
x = pandas.read_csv('cover.csv')
print(x)
#数据归一化处理
ss = StandardScaler()
scale_feature = ['col1','col2','col3','col4','col5','col6','col7','col8','col9']
x[scale_feature] = ss.fit_transform(x[scale_feature])

#Gaussian Naive Bayes
gnb = GaussianNB()
x_pred = gnb.fit(x[scale_feature],x['class']).predict(x[scale_feature])
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d" % (x[scale_feature].shape[0],(x[scale_feature] != x_pred).sum()))



# import pandas
# pandas.set_option('display.max_columns', None)
# print(x.describe(include='all'))
# x.hist(grid=False, figsize=(12,12))
# from sklearn import datasets
# iris = datasets.load_iris()
# print(iris)
