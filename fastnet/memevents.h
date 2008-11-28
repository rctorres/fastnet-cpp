/** 
@file  memevents.h
@brief MemEvents class declaration file.
*/


#ifndef MEMEVENTS_H
#define MEMEVENTS_H

#include <list>

#include "TrigRingerTools/fastnet/events.h"

using namespace std;


namespace FastNet
{
	/**
	 @brief		Class to store events in memory, for improved performance.
	 @author	Rodrigo Coura Torres (torres@lps.ufrj.br)
	 @version	1.0
	 @date		14/11/2004

	 Class to manage data events. This class works with the data fully loaded into memory, 
	 to improve speed, it stores the data in a linked list into memory, for fast data
	 access, and improved trainig/testing performance of neural networks. It should be
	 used when you have enough memory for holding the entire events to be presented to a 
	 network and training time is critical.
	 @see FastNet:Events
	*/
	class MemEvents : public Events
	{
		private:
			/// Linked list to store the evnts in memory.
			/**
			 This attribute is where the events will be stored to, since it is a linked list
			 stored in memory, the access is very fast, allowing reduced events accessing times.
			*/
			list<const REAL*> data;

			/// Linked list iterator.
			/**
			 This iterator is responsible for running through the linked list, and to get
			 the events as they are requested by the neural network.
			*/
			list<const REAL*>::const_iterator itr;

		public:
			//Inherited members.

			/// Gets an event from the linked list.
			/**
			 This method getos the event to where the internal iterator pinter is pointing to
			 After returning the event, the iterator pointer points to the next one in the list.
			 Before access an event, you must certify with hasNext if the list of events 
			 is not already finished.
			 @return The event pointed by the linked list iterator pointer.
			*/
			const REAL *readEvent()
			{
				return *itr++;
			}

			
			const REAL* readEvent(int evIndex)
			{
				//TODO
				return NULL;
			}
			
			
			const REAL* readRandomEvent(int &evIndex)
			{
				return NULL;
			}
			
			
			/// Points to the begining of the linked list.
			/**
			 This methos resets the linked list, by pointing its pointer to the first event
			 stored into it.
			*/
			void reset()
			{
				itr = data.begin();
			}


			/// Informs if there are more events to be read.
			/**
			 This method informs the calling function if there are still events to be read.
			 The function looks at the linked list iterator pointer and sees if it is pointing
			 to the end of the list.
			 @return true if there is another event to be read, false otherwise.
			*/
			bool hasNext() const
			{
				return (itr != data.end());
			}


			/// Default class constructor.
			/**
			 This constructor starts the class with an empty list.
			 So, in order to avoid errors, this method points the linked list pointer
			 to the end of the empty list, indicating that there is no data to be read.
			*/
			MemEvents()
			{
				itr = data.end();
			}


			/// Loading data class constructor.
			/**
			 This constructor initializes the linked list by reading the events from a file
			 After this constructor, if no errors occured, the class will have all the events
			 read from the input file loaded in the linked list.
			 @param fileName The name of the file where the events will be read from.
			 @param eventSize The size (number of variables) in each event.
			 @throw bad_alloc if an error occurs while allocating memory
			 @throw OPEN_FILE_ERROR in case of error reading the input file.
			*/
			MemEvents(const std::string &fileName, size_t eventSize)
			{
				try
				{
					open(fileName, eventSize);
					numInputs = eventSize;
					numEvents = list.size();
				}
				catch(bad_alloc xa)
				{
					throw;
				}
				catch(OPEN_FILE_ERROR xb)
				{
					throw;
				}
			}


			/// Load the events from a file.
			/**
			 This method initializes the linked list by reading the events from a file
			 After this, if no errors occured, the class will have all the events
			 read from the input file loaded in the linked list.
			 @param fileName The name of the file where the events will be read from.
			 @param eventSize The size (number of variables) in each event.
			 @throw bad_alloc if an error occurs while allocating memory
			 @throw OPEN_FILE_ERROR in case of error reading the input file.
			*/
			void open(const std::string &fileName, size_t eventSize)
			{
				const unsigned dataSize = eventSize * sizeof(REAL);
	
				ifstream in(fileName.c_str(), ios::binary);
				if (!in) throw "Impossible to open the event file!";

				try
				{
					//Reading an event at a time and storing in 'data'.
					REAL *event = new REAL [eventSize];
					in.read((char*) event, dataSize);

					while (!in.eof())
					{
						data.push_back(event);
						event = new REAL [eventSize];
						in.read((char*) event, dataSize);
					}

					reset();
				}
				catch(bad_alloc xa)
				{
					throw;
				}

				in.close();
			}


			/// Class destructor.
			/**
			 This destructor deallocates all the memory used to store the events
			 in the linked list.
			*/
			virtual ~MemEvents()
			{
				itr = data.begin();
				while(itr != data.end()) delete [] *itr++;
			}
	};
}

#endif
