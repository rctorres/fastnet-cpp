//Reporter.h

#ifndef REPORTER_H
#define REPORTER_H

#include <sstream>

namespace sys
{
  /**
   * Defines the interface to the Reporting System.
   *
   * This class defines how the messages and errors are sent to the
   * monitoring user. Since the implementation of this class can
   * change depending on the environment, we have decoupled from its
   * interface by using the <em>bridge</em> pattern.
   */
  class Reporter
  {
  public:
    
    /**
     * Destructor virtualisation.
     */
    virtual ~Reporter() {};

  public:

    /**
     * Report something to the user.
     * @param info What to report.
     */
    virtual void report (const std::string& info)
    {
      mexPrintf((info + "\n").c_str());
      mexEvalString("drawnow;");
    }


    /**
     * Warn the user about a problem.
     * @param info What to warn about.
     */
    virtual void warn (const std::string& info)
    {
      mexWarnMsgTxt((info + "\n").c_str());
      mexEvalString("drawnow;");
    }


    /**
     * Warn the user about a problem and std::exit() afterwards.
     * @param info What to report about the problem.
     */
    virtual void fatal (const std::string& info)
    {
      mexErrMsgTxt((info + "\n").c_str());
    }

    /**
     * Warn the user about an exception.
     * @param info What to report about the problem.
     */
    virtual void except (const std::string& info)
    {
      mexWarnMsgTxt((info + "\n").c_str());
      mexEvalString("drawnow;");
    }
  };
}

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
#define REPORT(r,m) {std::ostringstream s; s << m; r->report(s.str());}
#else
#define REPORT(r,m)
#endif

/**
 * Defines a simpler way to report messages
 */
#define FATAL(r,m) {std::ostringstream s; s << m; r->fatal(s.str());}

/**
 * Defines a simpler way to report messages
 */
#define EXCEPT(r,m) {std::ostringstream s; s << m; r->except(s.str());}

/**
 * Defines a simpler way to report messages
 */
#if (VERBOSE>=1)
#define WARN(r,m) {std::ostringstream s; s << m; r->warn(s.str());}
#else
#define WARN(r,m)
#endif

#endif /* REPORTER_H */ 
