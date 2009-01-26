//Reporter.h

#ifndef REPORTER_H
#define REPORTER_H

#include <sstream>

/**
 * Make sure that we have verbose on if the user does not specify
 * anything else. If VERBOSE is set to 1 or above, warnings
 * messages are printed. If above or equal 2, report messages are
 * also printed. Fatal and exceptions messages are always printed.
 */
#ifndef VERBOSE
#define VERBOSE 2
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

#if (DEBUG==0)
#define DEBUG0(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG1(m)
#define DEBUG2(m)
#elif (DEBUG==1)
#define DEBUG0(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG1(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG2(m)
#elif (DEBUG>=2)
#define DEBUG0(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG1(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#define DEBUG2(m){std::ostringstream s; s << m; mexPrintf((s.str() + "\n").c_str()); mexEvalString("drawnow;");}
#endif

#else //debug

#define DEBUG0(m)
#define DEBUG1(m)
#define DEBUG2(m)

#endif //DEBUG

#endif /* REPORTER_H */ 
