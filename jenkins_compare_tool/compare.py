import urllib
import re
from bs4 import BeautifulSoup
import sys

def generate_url(address,username,password):
    #print address
    #print username
    #print password
    return "https://{0}:{1}@{2}".format(username,password,address)

def get_page(address,username,password):
    url = generate_url(address,username,password)
    #print url
    f = urllib.urlopen(url)
    lines = f.read()
    f.close()
    return lines


def parse_page(content):
    soup = BeautifulSoup(content, "lxml")
    all = soup.findAll('td',{ "class" : "pane" })
    test_suite_list = []
    for i in all:
        line = i.find('a')
        if line:
            m = re.search('testsuite=.*', str(line.contents[0]))
            if m:
                if m.group().split("=")[1].startswith("NST"):
                    test_suite_list.append(m.group().split("=")[1])
    return test_suite_list

    
def compare_two_lists(l1,l2):
    result = {}
    for i in l1:
        result[i] = [1,0]

    for i in l2:
        if i in result:
            result[i] = [1, 1]
        else:
            result[i] = [0, 1]

    inl1notinl2 = filter(lambda x:x[1][1] == 0,result.items())
    inl2notinl1 = filter(lambda x:x[1][0] == 0,result.items())
    return result, inl1notinl2, inl2notinl1
    

def test_parse():
    address = "sasn-jenkins-pc.mo.sw.ericsson.se/view/PL_SASN_MAINTENANCE/view/MAINTENANCE_14A15B_BRANCH/job/PL_SCOPE_FT_FULL_15B/12/testReport/"
    username = "eyiilei"
    password = "Ly@sasn14"
    print parse_page(get_page(address,username,password))

def test_compare():
    l1 = [1,2,3]
    l2 = [2,3,4]
    result, el1, el2 = compare_two_lists(l1,l2)
    print result
    print el1
    print el2
    
def run(address1, address2, username, password):
    test_suite_list1 = parse_page(get_page(address1,username,password))
    test_suite_list2 = parse_page(get_page(address2,username,password))
    result, el1, el2 = compare_two_lists(test_suite_list1, test_suite_list2)
    print "in l1 but not in l2"
    for i in el1:
        print i[0]

    print "in l2 but not in l1"
    for i in el2:
        print i[0]


if __name__ == "__main__":
#    test_compare()
    username = sys.argv[1]
    password = sys.argv[2]
    address1 = sys.argv[3]
    address2 = sys.argv[4]
    
    run(address1, address2, username, password)


   

