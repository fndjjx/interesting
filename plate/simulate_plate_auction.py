import numpy as np
import random
import datetime


class people():
    def __init__(self,start,number):
        self.current_offer = 0
        self.price = start
        self.number = number
    def get_number(self):
        return self.number
    def get_current_offer(self):
        return self.current_offer
    def get_price(self):
        return self.price
    def add_current_offer(self):
        if self.current_offer >= 2:
            return False
        else:
            self.current_offer += 1
            return True
    def set_price(self,price):
        if self.add_current_offer():
            self.price = price
    def set_start_price(self,price):
        self.price = price
    

def random_result(left_time,left_count):
    r = random.random()
    ss = left_count/(float(left_time))
    if  r < (1-ss):
        return 0
    else:
        return 1

def simulate_once(member_count, start, quota):
    p = start
    people_dic = {}
    for i in range(member_count):
        people_dic[str(i)] = people(start,i)
    for i in range(60):
        offer = []
        cccc = 0
        for j in range(member_count):
            r = random_result(60-i,2-people_dic[str(j)].get_current_offer())
            pp = p
            people_dic[str(j)].set_start_price(p)
            if 0 == r:
                offer.append(pp)
            elif 1 == r:
                people_dic[str(j)].set_price(pp+300)
                offer.append(people_dic[str(j)].get_price())
            if people_dic[str(j)].get_current_offer()<2:
                cccc+=1
        print cccc/float(member_count)
        offer.sort(reverse = True)
        p = offer[quota-1]
        print p

    return p

def repeat(times, member_count, start_price, quota):
    r = []
    for i in range(times): 
        r.append(simulate_once(member_count, start_price, quota))

    print np.mean(r)
    print np.std(r)

if __name__ == "__main__":
    start = datetime.datetime.now() 
    repeat(1,221109,80600,8310)
    end = datetime.datetime.now() 
    interval = (end-start).seconds
    print interval
