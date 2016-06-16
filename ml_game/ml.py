from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import Perceptron
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier

import sys

def data_transform(datafile):

    x = []
    y = []
    with open(datafile) as f:
        lines = f.readlines()
        for eachline in lines:
            xx = []
            xx.append(int(eachline.split(" ")[1].split(":")[1]))
            xx.append(int(eachline.split(" ")[2].split(":")[1].split("\n")[0]))
            x.append(xx)
            y.append(int(eachline.split(" ")[0]))

    return (x,y)
        
        

class machine_learning():

    def __init__(self, data = None, isPCA = False, nu_pca = 0):
        
        if data:
            if isPCA:
                self.data = [self.pca_reduce(data[0], nu_pca), data[1]]
            else:
                self.data = self.load_data(data)
        else:
            self.data = self.load_iris_data()


    def load_iris_data(self):

        iris = datasets.load_iris()
        #print iris.data
        #print iris.target
        return (iris.data,iris.target)

    def load_data(self, data):
  
        return data


    def svm_train(self):
 
        #clf = svm.SVC()
        clf = svm.SVC(kernel='linear',probability = True)
        #clf = svm.LinearSVC(C=0.5,max_iter=10000)
        #clf = svm.SVC(kernel='poly',degree=3)
        clf.fit(self.data[0],self.data[1])

        return clf

    def rf_train(self):
        clf = RandomForestClassifier(n_estimators = 100)
        clf.fit(self.data[0],self.data[1])
        return clf


    
    def logistic_train(self):
    
        clf = LogisticRegression()
        clf.fit(self.data[0],self.data[1])
         
        return clf

    def neighbor_train(self):
        clf = neighbors.KNeighborsClassifier()
        clf.fit(self.data[0],self.data[1])

        return clf

    def bayes_train(self):
        clf = naive_bayes.GaussianNB()
        clf.fit(self.data[0],self.data[1])

        return clf

    def train(self, method):

        if method == "svm":
            clf = self.svm_train()
        elif method == "logistic":
            clf = self.logistic_train()
        elif method == "neighbor":
            clf = self.neighbor_train()
        elif method == "bayes":
            clf = self.bayes_train()
        elif method == "rf":
            clf = self.rf_train()

        return clf


    def predict(self, clf, test_data):

        return clf.predict(test_data)

    def predict_proba(self, clf, test_data):
        return clf.predict_proba(test_data)


    def kmeans_cluster(self, k):

        kmeans = KMeans(n_clusters = k)
        kmeans.fit(self.data[0])

        return kmeans

    def pca_reduce(self, data, k):

        reduced_data = PCA(n_components=k).fit_transform(data)
        return  reduced_data



    def judge(self, expect, predict):

        nu_tp = 0
        nu_fn = 0
        nu_fp = 0
        nu_tn = 0

        for i in range(len(expect)):
            if expect[i] == 1:
                if expect[i] == predict[i]:
                    nu_tp += 1
                elif expect[i] != predict[i]:
                    nu_fn += 1
            else:
                if expect[i] == predict[i]:
                    nu_tn += 1
                elif expect[i] != predict[i]:
                    nu_fp += 1

    #    print "tp:%s fn:%s fp:%s tn:%s"%(nu_tp, nu_fn, nu_fp, nu_tn)
        accuracy = float(nu_tp + nu_tn)/(nu_tp + nu_fn + nu_fp + nu_tn)
        precise = float(nu_tp)/(nu_tp + nu_fp)
        recall = float(nu_tp)/(nu_tp + nu_fn)

        return accuracy, precise, recall

        
      

if __name__ == "__main__":
#    python ml.py pca/nopca svm/logistic

#    train_data_x = [[1,2],[1,1],[2,3],[5,1],[6,1],[7,1]]
#    train_data_y = [1,1,1,0,0,0]
#    train_data = [train_data_x,train_data_y]
#    test_data = [10,2]
    data =  data_transform("train_data")
    train_data = [data[0][:-1000],data[1][:-1000]]
    test_data = [data[0][-1000:],data[1][-1000:]]



    if sys.argv[1] == "pca":
        ml = machine_learning(train_data, True, 1)
        test_data[0] = ml.pca_reduce(test_data[0], 1)
    elif sys.argv[1] == "nopca":
        ml = machine_learning(train_data)
    else:
        print "pca or not, choose one"
        exit()

    result = []
    if sys.argv[2] == "svm":
        for i in range(len(test_data[0])):
            single_test_data = test_data[0][i]
            result.append(ml.predict("svm", single_test_data)[0]) 
    elif sys.argv[2] == "logistic":
        for i in range(len(test_data[0])):
            single_test_data = test_data[0][i]
            result.append(ml.predict("logistic", single_test_data)[0]) 
    else:
        print "choose the right algorithm"
        exit()

    #print "expect:  %s "%test_data[1]
    #print "predict: %s "%result
    print ml.judge(test_data[1], result)
