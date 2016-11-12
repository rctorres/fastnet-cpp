//Reporter.h

#ifndef REPORTER_H
#define REPORTER_H

#include <iostream>

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
#define REPORT(m) {std::cout << m; std::cout.flush();}
#else
#define REPORT(m)
#endif

/**
 * Defines a simpler way to report messages
 */
#define FATAL(m){std::cout << m << std::endl;}

/**
 * Defines a simpler way to report messages
 */
#define EXCEPT(m){std::cout << m << std::endl; std::cout.flush();}

/**
 * Defines a simpler way to report messages
 */
#if (VERBOSE>=1)
#define WARN(m){std::cout << m << std::endl; std::cout.flush();}
#else
#define WARN(m)
#endif

#ifdef DEBUG

#if (DEBUG==1)
#define DEBUG1(m){std::cout << m << std::endl; std::cout.flush();}
#define DEBUG2(m)
#define DEBUG3(m)
#elif (DEBUG==2)
#define DEBUG1(m){std::cout << m << std::endl; std::cout.flush();}
#define DEBUG2(m){std::cout << m << std::endl; std::cout.flush();}
#define DEBUG3(m)
#elif (DEBUG>=3)
#define DEBUG1(m){std::cout << m << std::endl; std::cout.flush();}
#define DEBUG2(m){std::cout << m << std::endl; std::cout.flush();}
#define DEBUG3(m){std::cout << m << std::endl; std::cout.flush();}
#endif

#else //debug

#define DEBUG1(m)
#define DEBUG2(m)
#define DEBUG3(m)

#endif //DEBUG

#endif /* Reporter */ 
