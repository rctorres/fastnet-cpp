#ifndef MTHELPER_H
#define MTHELPER_H

#include <vector>

namespace MT
{
  inline void waitCond(bool &cond, pthread_mutex_t &mutex)
  {
    while (true)
    {
      pthread_mutex_lock(&mutex);
      if (cond)
      {
        pthread_mutex_unlock(&mutex);
        break;
      }
      pthread_mutex_unlock(&mutex);
    }
  }

  inline void safeSignal(bool &cond, pthread_mutex_t &mutex, pthread_cond_t &req)
  {
    waitCond(cond, mutex);
    pthread_cond_signal(&req);
  }

  inline void safeWait(bool &cond, pthread_mutex_t &mutex, pthread_cond_t &req)
  {
    pthread_mutex_lock(&mutex);
    cond = true;
    pthread_cond_wait(&req, &mutex);
    cond = false;
    pthread_mutex_unlock(&mutex);
  }
  
};

#endif
