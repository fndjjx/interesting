from ml import *
from sklearn import feature_selection 
from sklearn import lda
from sklearn import preprocessing
from sklearn import metrics
from sklearn import linear_model
import numpy as np
import copy
import pylab as pl


def format_dataset1(filename):
    with open(filename) as f:
        lines = f.readlines()
        target = []
        features = []
        for i in lines:
            if i.startswith("0") or i.startswith("1"):
                target.append(int(i.split(",")[0]))
                tmp = []
                tmp.append(int(i.split(",")[1]))
                features.append(tmp)


    return [features,target]


def solution_dataset1():
    train_data = format_dataset1("dataset/DataSet_1/DataSet_1_training.csv")
    test_data = format_dataset1("dataset/DataSet_1/DataSet_1_test.csv")
#    print train_data
#    print test_data
    ml = machine_learning(train_data)
    clf = ml.train("rf")
    result = []
    for i in range(len(test_data[0])):
        single_test_data = test_data[0][i]
        result.append(ml.predict(clf, single_test_data)[0])
    print "acc:%s pre:%s  recall:%s"%ml.judge(test_data[1], result)

    fpr, tpr, th = metrics.roc_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 1)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve')

    precision, recall, th = metrics.precision_recall_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')

#    pl.show()


def format_dataset2(filename):
    with open(filename) as f:
        lines = f.readlines()
        target = []
        features = []
        for i in lines:
            if i.startswith("0") or i.startswith("1"):
                target.append(int(i.split(",")[0]))
                tmp = []
                tmp.append(int(i.split(",")[1]))
                tmp.append(int(i.split(",")[2]))
                features.append(tmp)

    return [features,target]

def solution_dataset2():
    train_data = format_dataset2("dataset/DataSet_2/DataSet_2_training.csv")
    test_data = [features,target] =  format_dataset2("dataset/DataSet_2/DataSet_2_test.csv")
#    print train_data
#    print test_data
    ml = machine_learning(train_data)
    clf = ml.train("logistic")
    print clf.coef_
    print clf.intercept_

    result = []
    for i in range(len(test_data[0])):
        single_test_data = test_data[0][i]
        result.append(ml.predict(clf, single_test_data)[0])
    print "acc:%s pre:%s  recall:%s"%ml.judge(test_data[1], result)

#    fpr, tpr, th = metrics.roc_curve(test_data[1], result)
#    ax = pl.subplot(2, 1, 1)
#    ax.plot(fpr, tpr)
#    ax.set_title('ROC curve')
#
#    precision, recall, th = metrics.precision_recall_curve(test_data[1], result)
#    ax = pl.subplot(2, 1, 2)
#    ax.plot(recall, precision)
#    ax.set_ylim([0.0, 1.0])
#    ax.set_title('Precision recall curve')

#    pl.show()
    draw(np.array(features,dtype=np.float),np.array(result,dtype=np.float))

def draw(X,Y):

    h = 20
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    pl.figure(1, figsize=(4, 3))
    pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

    pl.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=pl.cm.Paired)

    pl.xlim(xx.min(), xx.max())
    pl.ylim(yy.min(), yy.max())
    pl.xticks(())
    pl.yticks(())
    pl.show()




def format_dataset3(filename):
    with open(filename) as f:
        lines = f.readlines()
        target = []
        features = []
        for i in lines:
            if i.startswith("0") or i.startswith("1"):
                target.append(int(i.split(",")[0]))
                tmp = []
                for j in range(1,5):
                    tmp.append(int(i.split(",")[j]))
                features.append(tmp)

    return [features,target]


def format_dataset4(filename):
    with open(filename) as f:
        lines = f.readlines()
        target = []
        features = []
        for i in lines:
            if i.startswith("0") or i.startswith("1"):
                target.append(int(i.split(",")[0]))
                tmp = []
                for j in range(1,7):
                    tmp.append(int(i.split(",")[j]))
                features.append(tmp)

    return [features,target]


def choose_feature(train_data, index):
    result = []
    index1 = []
    for i in index:
        if i:
            index1.append(1)
        else:
            index1.append(0)

    for i in train_data:
        tmp1 = (list(np.array(i)*np.array(index1)))
        tmp2 = []
        for j in tmp1:
            if j != 0 :
                tmp2.append(j)
        result.append(tmp2)

    return result


def pre_process1(data):

    d = np.array(data)
    dd = d.T

    mat = []
    for i in list(dd):
        row = []
        for j in i:
            row.append(float(j-min(i))/(max(i)-min(i)))
        mat.append(row)

    return list(np.array(mat).T)

        

def solution_dataset4():
    train_data = format_dataset4("dataset/DataSet_4/DataSet_4_training.csv")
    test_data = format_dataset4("dataset/DataSet_4/DataSet_4_test.csv")

#    print train_data
#    print test_data
    #ml = machine_learning(train_data,True,3)

    #train_data[0] = preprocessing.scale(train_data[0])
    #train_data[0] = preprocessing.normalize(train_data[0])
    #test_data[0] = preprocessing.normalize(test_data[0])
    #test_data[0] = preprocessing.scale(test_data[0])
    train_data1 = copy.deepcopy(train_data)
    train_data2 = copy.deepcopy(train_data)
    train_data3 = copy.deepcopy(train_data)
    train_data1[0] = np.array(train_data[0])[:,0:2]
    train_data2[0] = np.array(train_data[0])[:,2:4]
    train_data3[0] = np.array(train_data[0])[:,4:6]
    #train_data1[0] = pre_process(train_data[0])
    #train_data2[0] = pre_process(train_data[0])
    #train_data1[1] = train_data[1]
    #train_data2[1] = train_data[1]

    ml1 = machine_learning(train_data)
    clf1 = ml1.train("rf")
    #print clf
    #selector = feature_selection.RFE(clf1,6)
    #selector = selector.fit(train_data[0], train_data[1])
    
    #clf1 = selector    

    #test_data[0] = ml.pca_reduce(test_data[0],3)

    result1 = []
    for i in range(len(train_data[0])):
        single_test_data = train_data[0][i]
        result1.append(ml1.predict_proba(clf1, single_test_data)[0])

    ml2 = machine_learning(train_data)
    clf2 = ml2.train("logistic")
    result2 = []
    for i in range(len(train_data[0])):
        single_test_data = train_data[0][i]
        result2.append(ml2.predict_proba(clf2, single_test_data)[0])
#
#
#
    ml3 = machine_learning(train_data)
    clf3 = ml3.train("logistic")
    result3 = []
    for i in range(len(train_data[0])):
        single_test_data = train_data[0][i]
        result3.append(ml3.predict(clf3, single_test_data)[0])
#
#
#
    train_data = [zip(result1,result2),train_data[1]]    
    print train_data
#
#    ml = machine_learning(train_data)
#    #ml = machine_learning(tmp)
#    clf = ml.train("neighbor")
#
#
    result = []
    rr = []
    rr2 = []
    for i in range(len(test_data[0])):
        single_test_data = test_data[0][i]
        r1 = ml1.predict_proba(clf1,single_test_data)[0]
        r2 = ml2.predict_proba(clf2,single_test_data)[0]
        r3 = ml3.predict_proba(clf3,single_test_data)[0]
        #single_test_data = [r1, r2]
#        result.append(ml.predict(clf,single_test_data)[0])
        if ((r1[0]+r2[0])/(r1[1]+r2[1]))<0.75:
            result.append(1)
        else:
            result.append(0)
#        result.append(r1)

    print "final result" 
    print "acc: %spre:%s  recall:%s"%ml1.judge(test_data[1], result)

    fpr, tpr, th = metrics.roc_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 1)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve')

    precision, recall, th = metrics.precision_recall_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')

#    pl.show()

def solution_dataset3():
    train_data = format_dataset3("dataset/DataSet_3/DataSet_3_training.csv")
    test_data = format_dataset3("dataset/DataSet_3/DataSet_3_test.csv")


    ml1 = machine_learning(train_data)
    clf1 = ml1.train("rf")

    ml2 = machine_learning(train_data)
    clf2 = ml2.train("logistic")
#
#
#
#
    result = []
    for i in range(len(test_data[0])):
        single_test_data = test_data[0][i]
        r1 = ml1.predict_proba(clf1,single_test_data)[0]
        r2 = ml2.predict_proba(clf2,single_test_data)[0]
        #single_test_data = [r1, r2]
#        result.append(ml.predict(clf,single_test_data)[0])
        if ((r1[0]+r2[0])/(r1[1]+r2[1]))<0.65:
            result.append(1)
        else:
            result.append(0)
#        result.append(r1)

    print "final result"
    print "acc%s pre:%s  recall:%s"%ml1.judge(test_data[1], result)

    fpr, tpr, th = metrics.roc_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 1)
    ax.plot(fpr, tpr)
    ax.set_title('ROC curve')

    precision, recall, th = metrics.precision_recall_curve(test_data[1], result)
    ax = pl.subplot(2, 1, 2)
    ax.plot(recall, precision)
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Precision recall curve')
#    pl.show()

if __name__ == "__main__":

    choice = int(sys.argv[1])
    if choice == 1:
        solution_dataset1()
    elif choice == 2:
        solution_dataset2()
    elif choice == 3:
        solution_dataset3()
    elif choice == 4:
        solution_dataset4()

