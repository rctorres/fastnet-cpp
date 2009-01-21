import commands

import sc_globals

def matlabBuild(target, source, env, for_signature):
	if commands.getstatusoutput('mex -help')[0] != 0:
		print "\nWARINING: MATLAB Compiler not found! Skiping matlab modules...\n"
		return ''
	
	incPaths = ' '.join(['-I%s' % i for i in env['CPPPATH']])
	libPaths = ' '.join(['-L%s' % i for i in env['LIBPATH']])
	libList = ' '.join(['-l%s' % i for i in env['LIBS']])
	compFlags = ' '.join(env['CCFLAGS'])
	return 'mex -cxx %s %s %s %s -o %s %s' % (compFlags, incPaths, libPaths, libList, target[0], source[0]);


def getMatlabSuffix():
	retCode, ext = commands.getstatusoutput('mexext');
	if retCode != 0: return 'mex';
	else: return ext;


matlab = {}

matlab['ntrain'] = {}
matlab['ntrain']['LIBS'] = ['neuralnet']

matlab['nsim'] = {}
matlab['nsim']['LIBS'] = ['neuralnet']

matlab['nsim_mt'] = {}
matlab['nsim_mt']['LIBS'] = ['neuralnet', 'pthread']
