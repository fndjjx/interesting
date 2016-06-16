
import os
import sys
import subprocess
import time

class runtestsuite(object):
    
    def __init__(self,parser=None):
        self.parser = parser
        
    def add_options(self):
        self.parser.add_argument('-t',dest='testsuiteyml',required=True)
        self.parser.add_argument('-n',action='store_true',default=False)
        self.parser.add_argument('-b',action='store_true',default=False)

    def set_args(self,args):
        self.args = args

    def run(self,run_logdir):
        env_var = os.environ
        if self.args.n:
            env_var["NEW_WIZARD"] = "y"
        testsuite_yml = self.args.testsuiteyml
        run_logfile = run_logdir+"/runtestsuite_log"
        fp = open(run_logfile,'w')
        if self.args.b:
            proc = subprocess.Popen(["/opt/nstest/bin/nspytest",testsuite_yml],env=env_var,stdout=fp,stderr=fp)
        else:
            proc = subprocess.Popen(["/opt/nstest/bin/nspytest",testsuite_yml],env=env_var,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            for out in proc.stdout:
                sys.stdout.write(out)
                fp.write(out)
            for out in proc.stderr:
                sys.stdout.write(out)
                fp.write(out)
        fp.close()


    
        
