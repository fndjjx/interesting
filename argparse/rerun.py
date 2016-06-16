import subprocess
import os
import yaml

class reruntestsuite(object):
    
    def __init__(self,parser=None):
        self.parser = parser
        self.rerunlist = []
        self.log_dir = None
        self.caselist = []
    def add_options(self):
        self.parser.add_argument('-r',dest='result',required=True)
        self.parser.add_argument('-o',dest='originyml',required=True)
        self.parser.add_argument('status')

    def set_args(self,args):
        self.args = args
    
    def run(self,rerun_logdir):
        self.log_dir = rerun_logdir
        self.getcaselist(self.args.status)
        rerunyml = self.getrerunyml()
        cmd = "ftts runtestsuite -t "+rerunyml+" -b"
        proc = subprocess.Popen(['ftts','runtestsuite','-t',rerunyml,'-b'],shell=False,stdout=None) 

    def getcaselist(self,status):
        
        cmd1 = self.args.result
        cmd2 = '/.*Test Cases '+status+':.*/ {found=1;next} /^-[[:blank:]]TC/ { if(found) {print $0} } /^$/ {found=0}'
        cmd3 = 's/^- //g'
        proc1 = subprocess.Popen(["cat",cmd1],shell=False,stdout=subprocess.PIPE)
        proc2 = subprocess.Popen(["awk",cmd2],shell=False,stdin=proc1.stdout,stdout=subprocess.PIPE)
        proc3 = subprocess.Popen(["sed",cmd3],shell=False,stdin=proc2.stdout,stdout=subprocess.PIPE)
        ret_code = proc3.wait()
        for out in proc3.stdout:
            case = "/opt/nstest/test/"+self.args.originyml.split('/')[-2]+"/"+out.strip()+".yml"
            self.caselist.append(case)

    def getrerunyml(self):
        
        original_yaml_file = open(self.args.originyml,'r')
        original_yaml = yaml.load(original_yaml_file)
        original_yaml["test_cases_pattern_list"] = self.caselist
        tmp_yaml_file = self.log_dir+'/temp_yaml'
        fp = open(tmp_yaml_file,'w')
        yaml.dump(original_yaml,fp)
        fp.close()
        return tmp_yaml_file



