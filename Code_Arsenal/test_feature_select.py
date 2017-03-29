
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn import cluster
from sklearn import preprocessing


# 读入数据存放至data_df的数据结构中

# In[3]:

data_path = '/home/390672/WORKSHOP/noise_analysis/清洗后的特征数据/混合特征提取/6特征集全齐/normalfulldata_source.xlsx'

data_df = pd.read_excel(data_path)
data_df.index = range(data_df.shape[0])


# 一. 数据预处理工作

# In[4]:

# 如果某列属性整列都是nan值，直接将该列删除
data_df = data_df.dropna(how="all",axis=1)

columns = data_df.columns
# 对inf的数据进行填充
for each in columns:

    temp = data_df.loc[:,each]
    temp.loc[temp == np.inf] = np.nan
    data_df.ix[:,each] = temp


# In[5]:

def fullfill_nan(data_df,names):
    # input:
    # data_df - 数据表DataFrame结构，一张完整的数据表
    # names - 特征集的名称，list
    # output：
    # 将特征集中所有有效的特征（不含nan值）进行聚类，在根据聚类情况将值赋给nan的元素
    
    # 输入的names对应的特征集的所有数据
    cluster_data = data_df[names]
    # cluster_data2 - 只包含全部有效值数据表（完全没有nan值）
    cluster_data2 = cluster_data
    for each in names:
        if not(any(cluster_data2[each].isnull())):
            # 没有非空的值，全部是有效值
            pass
        else:
            cluster_data2 = cluster_data2.drop(each,axis=1)

    # col_valid 所有有效的特征的列名
    # col_rest 所有含无效值的特征的列名
    col_full = cluster_data.columns
    col_valid = cluster_data2.columns.values.tolist()
    col_rest = col_full.drop(col_valid).values.tolist()

    for each in col_rest:
        nan_ind = cluster_data2.index[data_df[each].isnull()]
        notnan_ind = cluster_data2.index[data_df[each].notnull()]

        train_data = cluster_data2.ix[notnan_ind,:] # 将nan的样本点剔除掉的样本库
        nan_data = cluster_data2.ix[nan_ind,:] # 只有nan的样本点的样本库
        try:
            ###################################################################################
            ################  对读入的数据进行聚类，再根据聚类结果填充空值 ####################
            ###################################################################################
            # 数据预处理（无量纲minmax）
            min_max_scaler = preprocessing.MinMaxScaler()
            X_train_minmax = min_max_scaler.fit_transform(train_data)
            X_nan_minmax = min_max_scaler.fit_transform(nan_data)
            # 进行聚类 K-Means
            km = cluster.KMeans(n_clusters=2).fit(X_train_minmax)
            labels = km.labels_
    
            notnan_ind_0 = train_data[labels == 0].index
            notnan_ind_1 = train_data[labels == 1].index
            mean_0 = data_df.ix[notnan_ind_0,each].mean()
            mean_1 = data_df.ix[notnan_ind_1,each].mean()
    
            nan_labels = km.predict(X_nan_minmax)
    
            nan_ind_0 = nan_data[nan_labels == 0].index
            nan_ind_1 = nan_data[nan_labels == 1].index
    
            data_df.loc[nan_ind_0,each] = mean_0
            data_df.loc[nan_ind_1,each] = mean_1
        except BaseException as e:
            print(each + '缺失值填充失败，出现如下错误：\n',e)
            # 因为每个属性都有缺失值，没办法进行聚类计算补充
            nan_ind = cluster_data2.index[data_df[each].isnull()]
            notnan_ind = cluster_data2.index[data_df[each].notnull()]
            mean = data_df.ix[notnan_ind,each].mean()
            data_df.loc[nan_ind,each] = mean
                    
    return data_df
    resurn
# In[6]:
# 画ROC曲线
def plot_ROC(y,y_score):
    # 二值化标签
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]
    
    fpr = dict();tpr = dict();roc_auc = dict()
    plt.figure()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                               ''.format(i, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()
 
# In[6]:
# 画混淆矩阵图 
def plot_confusion_matrix(y_pred,y_test,names,title='Confusion matrix', cmap=plt.cm.Greens):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label') 
 
# In[6]:

import time 
time_a = time.time()
list1 = dict() #缺失数量大于20 %
list2 = dict() #缺失数量少于等于20%

# 对于某列含有部分缺失值的数据进行处理
for each in columns:
    temp = data_df.loc[:,each].notnull() # temp为True代表是非空的元素
    
    if all(temp):
        # 没有非空的值，全部是有效值
        pass
    elif len(temp[temp==False])/len(temp) > 0.2:
        # 1. 缺失数量大于20 %
        #  如果含有inf值的元素，使用nan值来替换
        #    用KNN找出最近邻的值作填充
        # 获取该列所处的特征集名称
        m = re.findall('.*_p',each) # each:'cl1_p2'; m :'cl1_p'
        group_name = m[0][:-2]  # group_name: 'cl1'
        list1[group_name] = [each_b for each_b in pd.Series(columns.values) if group_name in each_b]
        
    else:
        # 2. 缺失数量少于等于20%
        #    用平均值
        # 找出 缺失数量大于20 % 的列
        # 获取该列所处的特征集名称
        m = re.findall('.*_p',each) # each:'cl1_p2'; m :'cl1_p'
        group_name = m[0][:-2]  # group_name: 'cl1'
        list2[group_name] = [each_b for each_b in pd.Series(columns.values) if group_name in each_b]
        
for each in list1.keys():
    data_df = fullfill_nan(data_df=data_df, names=list1[each])


print(time.time() - time_a)


# 将剩余的没处理好的nan值数据填充完整

# In[7]:

for each in list2.keys():
    columns = list2[each]
    for each_b in columns:
        _ = data_df[each_b].fillna(data_df[each_b].mean(),inplace = True)





# 对标签进行哑变量编码化处理

# In[9]:

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
data_label = data_df['machine_type']
# lists of feature-value 
feature_value = list()
for each in data_label:
    feature_value.append({'machine_type':each})
data_label = vec.fit_transform(feature_value).toarray()

 

# 先进行特征选择

# In[10]:

# 获取数据样本集
data_source = data_df[data_df.columns.values[:-1]]

# 先对数据进行无量纲化的预处理
mms = preprocessing.MinMaxScaler()
data_df_pre = mms.fit_transform(data_source)
data_df_pre = pd.DataFrame(data_df_pre,columns=data_df.columns.values[:-1])

# In[17]:
from sklearn.preprocessing import label_binarize
# 将string型的样本标签改成数值型的向量
y = np.zeros([len(data_label),1])
y[data_label[:,0]== 1] = 1
y[data_label[:,1]== 1] = 2 # 0正常，1异常，2整改后正常
y = y.ravel()

#y = label_binarize(y, classes=[0,1,2])
#n_classes = y.shape[1]

# In[ ]:
# 【败】 RFECV的方法
#from sklearn.svm import SVC
#from sklearn.feature_selection import RFECV
#svc = SVC(kernel='rbf',gamma=10)
#rfecv = RFECV(estimator=svc, step=1, cv=3,
#              scoring='accuracy')
#rfecv_af = rfecv.fit(data_df_pre,y)
#print(rfecv_af.ranking_)
#print('>>>>>>>>>>>>>>finish!<<<<<<<<<<<<<<<<<<<')


# In[ ]:
# 随机森林的方法获取重要性高的特征

from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators = 250, random_state=0)
forest.fit(data_df_pre,y)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]


percentile = 0.5
left_num = int(np.around(data_df_pre.shape[1]*percentile,decimals=0)) # 剩下排名在前百分之percentile的数
feature_importance = pd.DataFrame(data_df_pre.ix[:,indices[:left_num]]\
                        ,columns=data_df_pre.columns[indices[:left_num]])
print('>>>>>>>>>>>>>>finish!<<<<<<<<<<<<<<<<<<<')


# In[ ]:
# 用PCA再做降维

from sklearn.decomposition import PCA
pca = PCA(n_components=0.99) # 将被找出来的pca，对累计特征值贡献率达到n_components%
pca.fit(feature_importance)
pca.explained_variance_ratio_
feature_pca = pca.transform(feature_importance)

# In[ ]:
# 对数据分成测试集和训练集
from sklearn import cross_validation
from sklearn import svm
from sklearn.metrics import roc_curve, auc, confusion_matrix

rs = cross_validation.ShuffleSplit(len(y), n_iter=3,\
     test_size=.25, random_state=0)

# In[ ]:
# 用svm看分类效果,用ROC和confusion图来表示
for train_index,test_index in rs:
    X_train, X_test = feature_pca[train_index],feature_pca[test_index]
    y_train, y_test = y[train_index],y[test_index]
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_score = clf.decision_function(X_test)
    y_predict = clf.predict(X_test)
    
#names = ['normal','abnormal','after']
#plot_confusion_matrix(y_predict,y_test,names=names)
print('>>>>>>>>>>>>>>finish!<<<<<<<<<<<<<<<<<<<')   
#plot_ROC(y_test,y_score)


# In[ ]:
# 做交叉验证
clf = svm.SVC()
scores = cross_validation.cross_val_score(clf,feature_pca,y,cv=10)

from sklearn.grid_search import GridSearchCV
# 做网格搜索获取SVC的参数
C_range = np.logspace(-2,10,4)
gamma_range = np.logspace(-9,2,4)
param_grid = dict(gamma = gamma_range, C=C_range)
cv = rs
grid = GridSearchCV(svm.SVC(kernel = 'rbf'),param_grid=param_grid, cv=cv)
grid.fit(feature_pca, y)

print("the best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

f1 = feature_pca[:,0].transpose()
f2 = feature_pca[:,1].transpose()
C = grid.best_estimator_.C
gamma = grid.best_estimator_.gamma
clf = svm.SVC(C=C,gamma=gamma)
X_2d = np.array([f1,f2]).transpose()
clf.fit(X_2d,y)

# 结果可视化
# clf:输入的模型
# f1:特征1
# f2:特征2
# y:数据标签
def plot_Classification(clf,f1,f2,y,gamma,C):
    x_max = np.max(f1)+1;x_min = np.min(f1)-1
    y_max = np.max(f2)+1;y_min = np.min(f2)-1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max,300), np.linspace(y_min,y_max,300))
    
    plt.figure()
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)#!!只能分两类!!
    plt.tilte("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
              size='medium')
    
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(f1, f2, c=y, cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
    plt.show()