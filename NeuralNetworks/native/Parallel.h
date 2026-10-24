#ifndef PARALLEL_H
#define PARALLEL_H
/* Reads of a file at any offset from any thread and a small thread pool, for the natives of NeuralNetworks. */
#include <stdint.h>
#include <stdlib.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#ifdef _WIN32
typedef HANDLE Opening;
/* The file at path opened for reads at any offset from any thread, its size, 0 if it can't be opened. */
static Opening OpenedFile(const char* path, uint64_t* size)
{
	LARGE_INTEGER length;
	HANDLE file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, 0);
	if (file == INVALID_HANDLE_VALUE)
		return 0;
	*size = GetFileSizeEx(file, &length) ? (uint64_t)length.QuadPart : 0;
	return file;
}

static void Closed(Opening file) { CloseHandle(file); }

/* Reads bytes at offset of the file into target, threads read at the same time, in parts of at most 1 GB. */
static int ReadFrom(Opening file, uint64_t offset, unsigned char* target, uint64_t bytes)
{
	int isRead = 1;
	HANDLE event = CreateEventA(0, TRUE, FALSE, 0);
	for (uint64_t done = 0; isRead && done < bytes;)
	{
		DWORD part = bytes - done > 1073741824 ? 1073741824 : (DWORD)(bytes - done), read = 0;
		OVERLAPPED overlapped = { 0 };
		overlapped.Offset = (DWORD)(offset + done);
		overlapped.OffsetHigh = (DWORD)((offset + done) >> 32);
		overlapped.hEvent = event;
		int isStarted = ReadFile(file, target + done, part, 0, &overlapped) || GetLastError() == ERROR_IO_PENDING;
		isRead = isStarted && GetOverlappedResult(file, &overlapped, &read, TRUE) && read > 0;
		done += read;
	}
	CloseHandle(event);
	return isRead;
}
#else
typedef intptr_t Opening;
/* The file at path opened for reads (its descriptor + 1), its size, 0 if it can't be opened. */
static Opening OpenedFile(const char* path, uint64_t* size)
{
	int file = open(path, O_RDONLY);
	if (file < 0)
		return 0;
	off_t length = lseek(file, 0, SEEK_END);
	*size = length > 0 ? (uint64_t)length : 0;
	return file + 1;
}

static void Closed(Opening file) { close((int)file - 1); }

static int ReadFrom(Opening file, uint64_t offset, unsigned char* target, uint64_t bytes)
{
	ssize_t read = 1;
	for (uint64_t done = 0; read > 0 && done < bytes; done += (uint64_t)read)
		read = pread((int)file - 1, target + done, bytes - done, (off_t)(offset + done));
	return read > 0;
}
#endif

#ifdef _WIN32
typedef SRWLOCK Lock;
typedef CONDITION_VARIABLE Condition;
static Lock poolLock = SRWLOCK_INIT;
static Condition poolStarted = CONDITION_VARIABLE_INIT, poolDone = CONDITION_VARIABLE_INIT;
static void Locked(Lock* lock) { AcquireSRWLockExclusive(lock); }
static void Unlocked(Lock* lock) { ReleaseSRWLockExclusive(lock); }
static void Waited(Condition* condition, Lock* lock) { SleepConditionVariableSRW(condition, lock, INFINITE, 0); }
static void Woken(Condition* condition) { WakeAllConditionVariable(condition); }
static uint64_t Processors(void) { return GetActiveProcessorCount(ALL_PROCESSOR_GROUPS); }
#else
typedef pthread_mutex_t Lock;
typedef pthread_cond_t Condition;
static Lock poolLock = PTHREAD_MUTEX_INITIALIZER;
static Condition poolStarted = PTHREAD_COND_INITIALIZER, poolDone = PTHREAD_COND_INITIALIZER;
static void Locked(Lock* lock) { pthread_mutex_lock(lock); }
static void Unlocked(Lock* lock) { pthread_mutex_unlock(lock); }
static void Waited(Condition* condition, Lock* lock) { pthread_cond_wait(condition, lock); }
static void Woken(Condition* condition) { pthread_cond_broadcast(condition); }
static uint64_t Processors(void) { return (uint64_t)sysconf(_SC_NPROCESSORS_ONLN); }
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define Paused() _mm_pause()
#else
#define Paused() ((void)0)
#endif
enum { Spins = 20000 };
static uint64_t poolThreads = 0, poolCount = 0;
static atomic_uint_fast64_t poolGeneration, poolBusy, poolNext;
static void (*poolWork)(void*, uint64_t) = 0;
static void* poolContext = 0;

static void Worked(void)
{
	for (uint64_t item = atomic_fetch_add(&poolNext, 1); item < poolCount; item = atomic_fetch_add(&poolNext, 1))
		poolWork(poolContext, item);
}

/* Waits until counter is no longer value, spinning a while (jobs follow each other quickly) before sleeping. */
static void Awaited(atomic_uint_fast64_t* counter, uint64_t value, Condition* condition)
{
	for (int spin = 0; atomic_load(counter) == value && spin < Spins; spin++)
		Paused();
	Locked(&poolLock);
	while (atomic_load(counter) == value)
		Waited(condition, &poolLock);
	Unlocked(&poolLock);
}

static void Wake(Condition* condition)
{
	Locked(&poolLock);
	Woken(condition);
	Unlocked(&poolLock);
}

/* A thread of the pool: works on every job started after it, the last one done wakes the starter. */
static void Serve(void)
{
	for (uint64_t seen = 0;; seen++)
	{
		Awaited(&poolGeneration, seen, &poolStarted);
		Worked();
		if (atomic_fetch_sub(&poolBusy, 1) == 1)
			Wake(&poolDone);
	}
}

#ifdef _WIN32
static DWORD WINAPI Served(void* unused) { Serve(); return 0; }
static int Started(void)
{
	HANDLE thread = CreateThread(0, 0, Served, 0, 0, 0);
	return thread && CloseHandle(thread);
}
#else
static void* Served(void* unused) { Serve(); return 0; }
static int Started(void)
{
	pthread_t thread;
	return pthread_create(&thread, 0, Served, 0) == 0 && pthread_detach(thread) == 0;
}
#endif

/* Runs work for the items 0 to count on this thread and the pool's: STRICT_THREADS in all, else a quarter of the
   logical processors (experts of Qwen3.8-Flash-Next on a Threadripper 3990X: 16 threads 47, 32 threads 40, 64 threads
   58 and 128 threads 135 ms per token). */
static void Parallel(void (*work)(void*, uint64_t), void* context, uint64_t count)
{
	const char* wanted = poolThreads ? 0 : getenv("STRICT_THREADS");
	uint64_t threads = poolThreads ? 0 : wanted && atoi(wanted) > 0 ? (uint64_t)atoi(wanted) : Processors() / 4;
	for (poolThreads = poolThreads ? poolThreads : 1; poolThreads < threads && Started(); )
		poolThreads++;
	poolWork = work;
	poolContext = context;
	poolCount = count;
	atomic_store(&poolNext, 0);
	atomic_store(&poolBusy, poolThreads - 1);
	atomic_fetch_add(&poolGeneration, 1);
	Wake(&poolStarted);
	Worked();
	for (uint64_t busy = atomic_load(&poolBusy); busy > 0; busy = atomic_load(&poolBusy))
		Awaited(&poolBusy, busy, &poolDone);
}
#endif
