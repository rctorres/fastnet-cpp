/** 
@file  events.h
@brief Events class declaration.
*/

#ifndef EVENTS_H
#define EVENTS_H

#include "TrigRingerTools/fastnet/defines.h"

namespace FastNet
{
	/**
		@brief Base class for events data handling for a neural network.
		@author		Rodrigo Coura Torres (torres@lps.ufrj.br)
		@version	1.0
		@date		14/11/2004

		This class is a pure virtual class for events handling for a neural network.
		Every neural network within the namespace FastNet acess the input and target (if it is the case)
		events by calling the events from this class, so the class that inherits this superclass
		must implement these methods in order to access the events wherever they are (memory, file, etc). 
		Doing so, the neural network doesn't have any idea from where the events are coming 
		from (a file, a ethernet adapter, memory, etc). So, the class that inherits from this class
		is responsible for acessing the data wherever they are, and to pass them to the neural network
		by means of the methods of this class.
	*/
	class Events
	{
		protected:
			
			/// The number of elements in an event.
			/**
			 This number specifies the number of variables (dimension) 
			 of the events.
			*/
			unsigned numInputs;
			
			/// The total numer of events in the data set.
			/**
			 Holds the total number of events contained in the data set.
			*/
			unsigned numEvents;
		

		public:
			//Pure virtual members.
			
			/// Reads the next stored event.
			/**
			This method must be implemented in order to get the next event. The method
			should also, after returns the next event points to the next one in line.
			Since it returns a pointer, the class that implements this method should
			create the necessary space to hold the event that will be used by the calling function.
			@return A pointer to the event to be used.
			*/
			virtual const REAL* readEvent() = 0;
			
			
			/// Reads the next stored event.
			/**
			This method must be implemented in order to get an specific.
			This method is useful for cases where a randomic input event was choosen
			and you must take the corresponding output event.
			Since it returns a pointer, the class that implements this method should
			create the necessary spece to hold the event that will be used by the calling function.
			@param[in] evIndex The index of the event to be read.
			@return A pointer to the event to be used.
			*/
			virtual const REAL* readEvent(size_t evIndex) = 0;
			
			
			/// Reads a randomic stored event.
			/**
			This method must be implemented in order to get a randomic event. It
			is useful when the events must be read in a randomic manner. 
			Since it returns a pointer, the class that implements this method should
			create the necessary spece to hold the event that will be used by the calling function.
			@param[out] evIndex The randomly choosen index of the event returned.
			@return A pointer to the randomic event to be used.
			*/
			virtual const REAL* readRandomEvent(size_t &evIndex) = 0;
			
			
			/// Points to the begining of the events list.
			/**
			This method resets the events list. It must be implemented in order
			to go back to the begining of the list, or perform any functionality that
			allows the neural network to access more events.
			*/
			virtual void reset() = 0;


			/// Tells if there is another event in the list.
			/**
			This method must be implemented in such a way that the calling function 
			can know if there is another event in the list, or it is already at the end.
			@return true if there is another event, false otherwise.
			*/
			virtual bool hasNext() const = 0;


			/// Gets the number of events handled by the object.
			/**
			 Gets the number of events that are available for use.
			 @return The number of events available.
			*/
			virtual unsigned getNumEvents() const {return numEvents;}
			
			
			/// Gets the size (dimension) of the events.
			/**
			 Gets the dimension (number of variables) of the events.
			 @return The dimension (size) of the events.
			*/
			virtual unsigned getEventSize() const {return numInputs;}
	};
}

#endif
