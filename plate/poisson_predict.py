import numpy as np
from scipy import stats
import sys
import os

class pmf():

    def __init__(self, choice = None):
        self.values = {}
        if choice:
            for i in choice:
                self.set_prob(i,1.0/len(choice))

    def set_prob(self, name, prob):
        self.values[name] = prob 


    def show_prob(self, name):
        return self.values[name]

    def normalize(self):
        total = 0
        for i in self.values:
            total += self.values[i]

        for i in self.values:
            self.values[i] = self.values[i]/float(total)

    def multiply(self, name, prob):
        self.values[name] = self.values[name]*prob

    def update_by_data(self, data):
        for i in self.values:
            self.multiply(i, self.likehood(i,data))
        self.normalize()

    def likehood(self):
        pass

class poisson():

    def __init__(self, lamb):
        self.lamb = lamb

    def get_pro(self, k):
        return (np.exp(-self.lamb)*(self.lamb**k))/np.math.factorial(k)


def mle_poisson(data):
    return sum(data)/float(len(data))

class binary():
    def __init__(self, p, k, n):
        self.n = n
        self.k = k
        self.p = p

    def get_pro(self):
        cnk = (np.math.factorial(self.n))/(np.math.factorial(self.k)*np.math.factorial(self.n-self.k))
        return cnk*(self.p**self.k)*((1-self.p)**(self.n-self.k))

class lamb_kde():
    def __init__(self, sample):
        self.kde = stats.gaussian_kde(sample)

    def estimate(self, x):
        print x
        return self.kde.evaluate(x)

    def get_pro(self, d1, d2 ,d3):
        sample = [d1,d2,d3]
        print "sample {}".format(sample)
        
        p = pmf()
        d = self.estimate(sample)
        print "d {}".format(d)
        for i in range(len(d)):
            p.set_prob(d3[i],d[i])
        p.normalize()
        return p.values


def load_data(directory):
    file_list = os.listdir(directory)
    print file_list
    result1 = []
    result2 = []
    for f in file_list:
        f = directory + '/' + f
        with open(f) as fp:
            lines = fp.readlines()
        tmp = []
        for eachline in lines:
            tmp.append(float(eachline.strip()))
        r = []
        for i in range(1,len(tmp)):
            if tmp[i]!=tmp[i-1]:
                r.append(tmp[i]-tmp[i-1])
        result1.append(len(r))
        result2.append(r)

    return result1,result2
            

if __name__ == "__main__":

    current_ln = float(sys.argv[1])
    current_pn = float(sys.argv[2])


    #sample_ln = [7441,7531,7454,8727,7763,7514,7698,9409,8363]
    #sample_pn = [172205,166302,166939,165765,170995,169159,179133,187533,196470]
    sample_ln = [8727,7763,7514,7698,9409,8363]
    sample_pn = [165765,170995,169159,179133,187533,196470]
    sample_count, sample_jump_amp = load_data("/home/ly/git_repo/my_program/interesting/plate/data")
    sample = [sample_ln,sample_pn,sample_count]
    print sample
    prior_estimat_model = lamb_kde(sample)

    count_prior = prior_estimat_model.get_pro([current_ln for i in range(30)], [current_pn for i in range(30)],[i for i in range(30)])
     

    lamb_mle = mle_poisson(sample_count)
    poisson_model = poisson(lamb_mle)
    count_likehood = [(i,poisson_model.get_pro(i)) for i in range(30)]
    mypmf = pmf()
    for i in count_likehood:
        mypmf.set_prob(i[0],i[1])

    print mypmf.values
    print count_prior

    pp = pmf()
    for i in range(30):
        d1 = mypmf.values[i]
        d2 = count_prior[i]
        d2 = 1
        pp.set_prob(i,d1*d2)

    pp.normalize()

    print "pp {}".format(type(pp.values))
    print sample_jump_amp

    pro_sum = 0
    pplist = []
    for k,v in pp.values.iteritems():
        pplist.append((k,v))

    pplist.sort(key=lambda x:x[0])
    print pplist
    ppsum=0
    for i in pplist:
        ppsum += i[1]
        if ppsum>0.95:
            break
    print i[0]
    print i[0]*100
    
        

