#!/usr/bin/python

import argparse
import os
import yaml
import time

class project(object):

    name = os.path.dirname(__file__)
    path = os.path.abspath(name)
    COMMANDS_FILE = path + '/config/commands.yml'
    ftts_logdirectory_prefix = '/tmp/ftts_log'
    def __init__(self):
        self.main_parser = None
        self.sub_parser = None
        self.sub_parsers = {} 
        self.sub_objects = {}
        self.COMMANDS = {}

        self.init_command()
        self.args = self.main_parser.parse_args()
    def init_command(self):
        self.load_command_config()
        self.add_command()
        self.add_object()
        self.add_command_options()
    def add_command(self):
        self.main_parser = argparse.ArgumentParser()
        self.sub_parser = self.main_parser.add_subparsers(dest="subcommand")
        for cmd in self.COMMANDS.keys():
            self.sub_parsers[cmd] = self.sub_parser.add_parser(cmd)
    def add_object(self):
        for cmd,object in self.COMMANDS.items():
            mod = __import__(cmd)
            command_object = getattr(mod,object)
            self.sub_objects[cmd] = command_object(self.sub_parsers[cmd])
    def add_command_options(self):
        for cmd in self.sub_objects.keys():
            self.sub_objects[cmd].add_options()
    def run(self):
        run_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        commandlogdir = self.ftts_logdirectory_prefix+"/"+run_time+"/"+self.args.subcommand
        os.makedirs(commandlogdir)
        self.sub_objects[self.args.subcommand].set_args(self.args)
        self.sub_objects[self.args.subcommand].run(commandlogdir)
    def load_command_config(self):
        file = open(self.COMMANDS_FILE,'r')
        self.COMMANDS = yaml.load(file)

def main():
    pro = project()
    pro.run()
if __name__ == "__main__":
    main()
