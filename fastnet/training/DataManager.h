#ifndef DATAMANAGER_H_H
#define DATAMANAGER_H_H

class DataManager
{
protected:
  unsigned pos;
  unsigned evSize;
  vector<REAL *> vec;
  
  
public:
  DataManager()
  {
    pos = 0;
    evSize = 0;
  }
  
  unsigned numEvents() const
  {
    return vec.size();
  }
  
  unsigned eventSize() const
  {
    return evSize;
  }
  
  unsigned getNextEventIndex()
  {
    if (pos == vec.size())
    {
      random_shuffle(vec.begin(), vec.end());
      pos = 0;
    }
    return pos++;
  }
  
  const REAL* operator[](unsigned idx) const
  {
    return vec[idx];
  };  
};

#endif
