#ifndef THREAD_STATE_H
#define THREAD_STATE_H
#include <stdlib.h>
static void ThreadReleased(void);
static _Thread_local int threadReady;
#ifdef _WIN32
#include <windows.h>
static INIT_ONCE threadOnce = INIT_ONCE_STATIC_INIT;
static DWORD threadSlot;
static void CALLBACK ThreadEnded(void* unused) { ThreadReleased(); }
static BOOL CALLBACK ThreadStarted(PINIT_ONCE once, PVOID parameter, PVOID* context)
{
	threadSlot = FlsAlloc(ThreadEnded);
	return threadSlot != FLS_OUT_OF_INDEXES;
}
static int ThreadReady(void)
{
	if (threadReady)
		return 1;
	if (!InitOnceExecuteOnce(&threadOnce, ThreadStarted, 0, 0) || !FlsSetValue(threadSlot, (void*)1))
		return 0;
	return threadReady = 1;
}
#else
#include <pthread.h>
static pthread_once_t threadOnce = PTHREAD_ONCE_INIT;
static pthread_key_t threadSlot;
static int threadStarted;
static void ThreadEnded(void* unused) { ThreadReleased(); }
static void ThreadStarted(void) { threadStarted = pthread_key_create(&threadSlot, ThreadEnded) == 0; }
static int ThreadReady(void)
{
	if (threadReady)
		return 1;
	if (pthread_once(&threadOnce, ThreadStarted) || !threadStarted || pthread_setspecific(threadSlot, (void*)1))
		return 0;
	return threadReady = 1;
}
#endif
#endif
