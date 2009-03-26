#ifndef MTHELPER_H
#define MTHELPER_H

#include <vector>

namespace MT
{
  pthread_cond_t trnProcRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valProcRequest = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t trnProcMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valProcMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t trnGetResRequest = PTHREAD_COND_INITIALIZER;
  pthread_cond_t valGetResRequest = PTHREAD_COND_INITIALIZER;
  pthread_mutex_t trnGetResMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t valGetResMutex = PTHREAD_MUTEX_INITIALIZER;

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
