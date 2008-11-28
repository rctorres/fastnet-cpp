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

#Am I using a MAC computer? Then I apply some optimizations for it
if 'Darwin' in platform.system():
  genCPPFlags += ['-fast']
  
incPath = ['../']
libPath = ['./']
