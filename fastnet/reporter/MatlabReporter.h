//Dear emacs, this is -*- c++ -*-

/**
 * @file MatlabReporter.h
 *
 * @brief This file contains the definition of what is a
 * sys::MatlabReporter. 
 */

#ifndef RINGER_SYS_MATLABREPORTER_H
#define RINGER_SYS_MATLABREPORTER_H

#include <iostream>
#include <string>

#include "fastnet/reporter/Reporter.h"
#include "mex.h"

namespace sys {

	/**
	* Defines a way to talk to a user that is sitting locally with
	* respect to the executing program.
	*/
	class MatlabReporter : public Reporter
	{

	public:

		/**
		* The only constructor.
		* @param os The <em>normal</em> output stream
		* @param es The error stream
		*/
		MatlabReporter(){}
			
		virtual ~MatlabReporter() {}
		
		bool report (const std::string& info)
		{
			mexPrintf((info + "\n").c_str());
			mexEvalString("drawnow;");
			return true;
		}
		
		bool warn (const std::string& info)
		{
			mexWarnMsgTxt((info + "\n").c_str());
			mexEvalString("drawnow;");
			return true;
		}

		bool fatal (const std::string& info)
		{
			mexErrMsgTxt((info + "\n").c_str());
			return true;
		}

		bool except (const std::string& info)
		{
			mexWarnMsgTxt((info + "\n").c_str());
			mexEvalString("drawnow;");
			return true;
		}

	};

}

#endif /* RINGER_SYS_MATLABREPORTER_H */
