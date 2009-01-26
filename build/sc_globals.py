import re
import os
import commands
import sys
import platform


def getSourceFiles(sourcesDir):
	srcFilter = re.compile('\.cxx\Z|\.cpp\Z|\.c\Z')
	return ['%s%s%s' % (sourcesDir, os.path.sep, f) for f in os.listdir(sourcesDir) if srcFilter.search(f) is not None]

#This is the global compiling flags list.
genCPPFlags = []

#Uncomment the -g line for optimized code.
debugFlag = ['-DDEBUG=2' '-g']
#debugFlag = []

#This is for finding the MATLAB include and library files withour the mex compiler.
#You should change this directory accordly so that the BOOST and MATLAB header and lib
#files can be found.
matlabIncPath = ['/Applications/MATLAB_R2008a/extern/include']
matlabLibPath = ['/Applications/MATLAB_R2008a/bin/maci']

#Am I using a MAC computer? Then I apply some optimizations for it
if 'Darwin' in platform.system():
  genCPPFlags += ['-fast']
  matlabIncPath = ['/Applications/MATLAB_R2008a/extern/include']
  matlabLibPath = ['/Applications/MATLAB_R2008a/bin/maci']
  
incPath = ['../'] + matlabIncPath
libPath = ['./'] + matlabLibPath
