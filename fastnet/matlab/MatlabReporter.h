//MatlabReporter.h

#ifndef MATLABREPORTER_H
#define MATLABREPORTER_H

#include <sstream>
#include <mex.h>

/**
 * Make sure that we have verbose on if the user does not specify
 * anything else. If VERBOSE is set to 1 or above, warnings
 * messages are printed. If above or equal 2, report messages are
 * also printed. Fatal and exceptions messages are always printed.
 */
#ifndef VERBOSE
#define VERBOSE 5
#endif

/**
 * Defines a simpler way to report messages
 */
#if (VERBOSE>=2) 
#define REPORT(m) {std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#else
#define REPORT(m)
#endif

/**
 * Defines a simpler way to report messages
 */
#define FATAL(m){std::ostringstream s; s << m; mexErrMsgTxt((s.str() + "\n").c_str());}

/**
 * Defines a simpler way to report messages
 */
#define EXCEPT(m){std::ostringstream s; s << m; mexWarnMsgTxt((s.str() + "\n").c_str()); mexEvalString("drawnow;");}

/**
 * Defines a simpler way to report messages
 */
#if (VERBOSE>=1)
#define WARN(m){std::ostringstream s; s << m; mexWarnMsgTxt((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#else
#define WARN(m)
#endif

#ifdef DEBUG

#if (DEBUG==1)
#define DEBUG1(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG2(m)
#define DEBUG3(m)
#elif (DEBUG==2)
#define DEBUG1(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG2(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG3(m)
#elif (DEBUG>=3)
#define DEBUG1(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG2(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG3(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#endif

#else //debug

#define DEBUG1(m)
#define DEBUG2(m)
#define DEBUG3(m)

#endif //DEBUG

#endif /* MatlabReporter */ 
