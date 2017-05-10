import sys
import os

"""
A class for logging data
(Before writing this I simply used nohup)
Adapted from: http://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting

Example:
	Setting
    sys.stdout = Logger()

    will make every print operation print to the log file and console
    
"""

class Logger(object):
    def __init__(self,log_file="log1.log"):

        self.terminal = sys.stdout
        self.base_dir = os.path.split(log_file)[0]
        try:
            os.makedirs(self.base_dir)
        except OSError:
            if not os.path.isdir(self.base_dir):
                raise

        
        self.log_file = log_file
        self.log = open(self.log_file, "a")
        print "Saving log to",self.log_file
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def close(self):
        self.log.close()
        
    def flush(self):
    #this flush method is needed for python 3 compatibility.
        pass

