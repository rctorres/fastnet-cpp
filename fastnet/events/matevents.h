/** 
@file  matevents.h
@brief MatEvents class declaration/definition file.
*/


#ifndef MATEVENTS_H
#define MATEVENTS_H

#include <algorithm>
#include <mex.h>

#include "fastnet/events/events.h"
#include "fastnet/defines.h"

using namespace std;

namespace FastNet
{
  /**
   @brief    Class to access data from the matlab environment.
   @author  Rodrigo Coura Torres (torres@lps.ufrj.br)
   @version  1.0
   @date    03/06/2005

   This class is used to access data that are generated in the Matlab environment
   So, by this class, the neural network can have access to training/testing
   events generated on that interface.
   @see FastNet:Events
  */
  class MatEvents : public Events
  {
    private:
    
      /// The index to the current event.
      /**
        Points to the current event to be accessed by the
        neural network.
      */
      size_t evCounter;
      
      /// Contains a usable copy of the current event.
      /**
       Since the events are stored in the Matrix class, and the
       neural network access them by a pointer to REAL, the current
       event to be accessed must be copyed to this kind of data holder,
       so, this pointer points to the place where a copy of the current 
       event is.
      */
      REAL *currEvent;
      
      /// Contains the list of scrambled indexes for ramdom event access.
      /**
       This vector contains the index of all events, ramdomly placed, so
       running sequentially this vector, using its values as index for the
       events vector assures a fast way to access the events in a ramdom way.
      */
      size_t *rndList;

      /// Contains the index of the next ramdom access event to be read.
      /**
       This variable holds the index of the next event to be read. Since it 
       get its value from the rndList vector, it always contains a ramdomly choosed
       event index.
      */
      size_t rndIndex;
      
      /// Typedef to deal with the funciont pointer.
      typedef  void (MatEvents::*FILL_FUNC_PTR)(const size_t);
      
      /// Holds the input data type (single or double).
      mxClassID dataType;
      
      /// Pointer to access the data in float format.
      float *fptr;
      
      /// Pointer to access the data in double format.
      double *dptr;
      
      /// Function pointer to address the correct reading format (single or double).
      FILL_FUNC_PTR getData;
    
      /// This is the function to be used for reading events in float format. 
      void readFloatValues(const size_t evNum)
      {
        size_t i,k;
        for (i = evNum*numInputs, k = 0; k<numInputs; i++, k++)
        {
          currEvent[k] = static_cast<REAL>(fptr[i]);
        }
      }

      /// This is the function to be used for reading events in float format.
      void readDoubleValues(const size_t evNum)
      {
        size_t i,k;
        for (i = evNum*numInputs, k = 0; k<numInputs; i++, k++)
        {
          currEvent[k] = static_cast<REAL>(dptr[i]);
        }
      }

    
    public:
    
      /// Class constructor.
      /**
       Initializes the event counter and data limits.
       @param[in] MatlabEvents A reference to the matrix passed by Matlab.
       @throw bad_alloc on memory allocation error.
       @throw DATA_INIT_ERROR on matrix limits error.
      */
      MatEvents(const mxArray *matlabEvents)
      {
        currEvent = NULL;
        rndList = NULL;
        fptr = NULL;
        dptr = NULL;
        rndIndex = 0;
        evCounter = 0;
        numInputs = static_cast<size_t>(mxGetM(matlabEvents));
        numEvents = static_cast<size_t>(mxGetN(matlabEvents));
        
        dataType = mxGetClassID(matlabEvents);
        if (dataType ==  mxSINGLE_CLASS)
        {
          fptr = static_cast<float*>(mxGetData(matlabEvents));
          getData = &MatEvents::readFloatValues;
        }
        else if (dataType ==  mxDOUBLE_CLASS)
        {
          dptr = static_cast<double*>(mxGetData(matlabEvents));
          getData = &MatEvents::readDoubleValues;
        }
        else throw "Data must be either float (single) or double!";
        
        // Checking if the events matrix is valid.      
        if ( (!numInputs) || (!numEvents) ) throw "Error in matrix limits!";

        try
        {
          currEvent = new REAL [numInputs];
          rndList = new size_t [numEvents];
          
          //Initializing the random list with values
          // and scrambling them.
          for (size_t i=0; i<numEvents; i++) rndList[i] = i;
          random_shuffle(rndList, (rndList+numEvents));
        }
        catch(bad_alloc xa)
        {
          throw;
        }
      }
      
      
      /// Class destructor.
      /**
       Releases the dynamically allocated memory used by the class.
      */
      ~MatEvents()
      {
        if (currEvent) delete [] currEvent;
        if (rndList) delete [] rndList;
      }
    
    
      /// Get the next event in the data source.
      /**
       Takes from matlab variable the next event to be accessed.
       The contents of that event is then copyed to a standart
       vector, to be accessed by the neural network. Before access
       an event, you must certify with hasNext if the list of events
       is not already finished.
       @return A pointer to the event to be used.
      */
      const REAL* readEvent()
      {
        ((this)->*(getData))(evCounter);
        evCounter++;
        return currEvent;
      }
      
      /// Reads an specific event.
      /**
       This method returns the event with the index specified by the calling function.
       @param[in] evIndex The index of the event to be returned.
       @return A pointer to the requested event.
      */
      const REAL* readEvent(const size_t evIndex)
      {
        ((this)->*(getData))(evIndex);
        return currEvent;
      }
      
      /// Read a ramdomly choosen event.
      /**
       Returns a ramdomly choosen event. This method also returns the index of the
       event returned, so the calling function will know which event was returned.
       @param[out] evIndex The index of the event returned.
       @return A pointer to the event ramdomly choosen.
      */
      const REAL* readRandomEvent(size_t &evIndex)
      {
        //If we reach the end of the list, we scramble the values again.
        if (rndIndex == numEvents)
        {
          random_shuffle(rndList, (rndList+numEvents));
          rndIndex = 0;
        }
        
        evIndex = rndList[rndIndex++];
        return readEvent(evIndex);
      }
      
      
      ///Resets the events list.
      /**
       Resets the events list, by going to the begining of
       the data set, so all the events can be read again.
      */
      void reset()
      {
        evCounter = 0;
      }


      ///Verifies if the list is not finished.
      /**
       Verifies if the data set still has events to present.
       @return true if there are still events in the set, false otherwise.
      */ 
      bool hasNext() const
      {
        if (evCounter < numEvents) return true;
        return false;
      }
  };
}

#endif
