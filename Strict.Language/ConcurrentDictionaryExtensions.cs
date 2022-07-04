using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Strict.Language;

//TODO: delete, now replaced by LazyCache
public static class ConcurrentDictionaryExtensions
{
	/// <summary>
	/// https://stackoverflow.com/a/65694270
	/// Returns an existing task from the concurrent dictionary, or adds a new task
	/// using the specified asynchronous factory method. Concurrent invocations for
	/// the same key are prevented, unless the task is removed before the completion
	/// of the delegate. Failed tasks are evicted from the concurrent dictionary.
	/// </summary>
	public static Task<Value> GetOrAddAsync<Key, Value>(
		this ConcurrentDictionary<Key, Task<Value>> source, Key key,
		Func<Key, Task<Value>> valueFactory) where Key : notnull =>
		source.TryGetValue(key, out var currentTask)
			? currentTask
			: CreateOnlyOnce(source, key, valueFactory);

	// ReSharper disable once MethodTooLong
	private static Task<Value> CreateOnlyOnce<Key, Value>(ConcurrentDictionary<Key, Task<Value>> source, Key key,
		Func<Key, Task<Value>> valueFactory) where Key : notnull
	{
		Task<Value>? newTask = null;
		var wrappedNewTask = new Task<Task<Value>>(async () =>
		{
			try
			{
				return await valueFactory(key).ConfigureAwait(false);
			}
			catch
			{
				// ReSharper disable AccessToModifiedClosure
				if (newTask != null)
					source.TryRemove(KeyValuePair.Create(key, newTask));
				throw;
			}
		});
		newTask = wrappedNewTask.Unwrap();
		var currentTask = source.GetOrAdd(key, newTask);
		if (currentTask == newTask)
			wrappedNewTask.RunSynchronously(TaskScheduler.Default);
		return currentTask;
	}
}