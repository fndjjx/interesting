import subprocess
import os
import yaml

class runtestcase(object):
    
    def __init__(self,parser=None):
        self.parser = parser
        self.log_dir = None
        self.caselist = []

    def add_options(self):
        self.parser.add_argument('-c',dest='testcases',required=True,nargs='*')
        self.parser.add_argument('-o',dest='originyml',required=True)

    def set_args(self,args):
        self.args = args
    
    def run(self,runcase_logdir):
        self.log_dir = runcase_logdir
        runcaseyml = self.getruncaseyml()
        #cmd = "ftts runtestsuite -t "+runcaseyml+" -b"
        proc = subprocess.Popen(['ftts','runtestsuite','-t',runcaseyml,'-b'],shell=False,stdout=None) 


    def getruncaseyml(self):
        
        original_yaml_file = open(self.args.originyml,'r')
        original_yaml = yaml.load(original_yaml_file)
        for case in self.args.testcases:
            self.caselist.append("/opt/nstest/test/"+self.args.originyml.split('/')[-2]+"/"+case.strip()+".yml") 
        original_yaml["test_cases_pattern_list"] = self.caselist
        tmp_yaml_file = self.log_dir+'/temp_yaml'
        fp = open(tmp_yaml_file,'w')
        yaml.dump(original_yaml,fp)
        fp.close()
        return tmp_yaml_file



