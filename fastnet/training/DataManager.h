#ifndef DATAMANAGER_H_H
#define DATAMANAGER_H_H

#include <vector>

class DataManager
{
protected:
  unsigned evSize;
  std::vector<REAL *> data;
  std::vector<unsigned> idx;
  std::vector<unsigned>::const_iterator nextEvent;
  
  void init(const unsigned numEvents)
  {
    DEBUG1("Initializing ramdom selector for " << numEvents << "events.")
    for (unsigned i=0; i<numEvents; i++) idx.push_back(i);
    random_shuffle(idx.begin(), idx.end());
    nextEvent = idx.begin();
  }

public:
  DataManager()
  {
    evSize = 0;
  }
  
  unsigned numEvents() const
  {
    return data.size();
  }
  
  unsigned eventSize() const
  {
    return evSize;
  }
  
  unsigned getNextEventIndex()
  {
    if (nextEvent == idx.end())
    {
      random_shuffle(idx.begin(), idx.end());
      nextEvent = idx.begin();
    }
    return *nextEvent++;
  }
  
  const REAL* operator[](const unsigned idx) const
  {
    return data[idx];
  };  
};

#endif
