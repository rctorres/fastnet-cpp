/** 
@file  defines.h
@brief Constants file.

  This file contains all the constants (defines and typedefs) used in the program.
*/

#ifndef DEFINES_H
#define DEFINES_H

#include <mex.h>
#include <string>

/// Specifies the version of the FastNet package.
/**
 This define must be used every time the FastNet package version must be presented.
*/
const std::string FASTNET_VERSION = "1.00";


///Implements a very simple exception class for file opening errors.
/** 
This typedef is used within a try/catch block to report a 
file opening error.
*/
typedef const char* OPEN_FILE_ERROR;

///Implements a very simple exception class for data initialization errors.
/** 
This typedef is used within a try/catch block to report data initialization errors.
*/
typedef const char* DATA_INIT_ERROR;

/// The file open error exception message.
/**
This message should be presented when a open file error exception occurs.
*/
const std::string OPEN_FILE_ERROR_MSG = "Impossible to open one or more files!";


///Default size for vectors containing strings.
/**
This constant should be used when a vector that will hold long strings,
like file names, for instance, must be created. 
*/
#define LINE_SIZE 500


/// Default value for small general use vectors.
#define SIZE 20


/// Default floating point word size.
/**
This data type must be used in ALL floating point variable declaration,
so, by simply changing its value we can easily change thw word size of all
floating point variables created. If you change this value, you *MUST* change the value
of REAL_TYPE as well.
*/
typedef double REAL;

///Sets the corresponding matlab type we will be using.
/**
This constant shoud be set to represent the same data type as REAL typedef. Ex: if REAL is
set to double, this constant should be changed to mxDOUBLE_CLASS.  If you change this value, 
you *MUST* change the value of REAL typedef as well.
*/
const mxClassID REAL_TYPE = mxDOUBLE_CLASS;


/// Macro to call pointers to member functions.
/**
Use this macro to call pointer to member functions. In order to improve speed
the transfer functions are stored in pointer to functions, and this macro makes
easy to call these pointers.
@param[in] ptrToTrfFunc pointer pointing to the member function you want to call.
*/
#define CALL_TRF_FUNC(ptrToTrfFunc)  ((this)->*(ptrToTrfFunc))


/// String ID for the hyperbolic tangent transfer function.
/**
This is the only ID for the hyperbolic tangent function for files, so, every time
that a file wants to make a reference that it will use this
function, this reference is done by this value.
*/
const std::string TGH_ID = "tansig";


/// String ID for the linear transfer function.
/**
This is the only ID for the linear transfer function for files, so, every time
that a file wants to make a reference that it will use this
function, this reference is done by this value.
*/
const std::string LIN_ID = "purelin";


/// String ID for the Gradient Descendent Backpropagation neural training.
/**
This is the only ID for the Gradient Descendent Backpropagation neural training for files, so, every time
that a file wants to make a reference that it will use this
training, this reference is done by this value.
*/
const std::string TRAINGD_ID = "traingd";


/// String ID for the Resilient Backpropagation neural training.
/**
This is the only ID for the Resilient Backpropagation neural training for files, so, every time
that a file wants to make a reference that it will use this
training, this reference is done by this value.
*/
const std::string TRAINRP_ID = "trainrp";


/// String ID used to inform that no value has been supplied.
const std::string NONE_ID = "NONE";


/// Macro to calculate the square of a number.
/**
@param[in] x the value which the square value you want to evaluate.
*/
#define SQR(x) ((x)*(x))


#endif
