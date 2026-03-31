using System.Reflection;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

/// <summary>
/// Searches for native implementations of trait methods in DLLs located in the same directory
/// as the .strict file being executed. When a .strict type declares a trait method (no body),
/// the runtime can search loaded DLLs for a matching type and method name to invoke natively.
/// </summary>
public static class NativePluginLoader
{
	public static object? TryCallNativeMethod(string typeName, string methodName,
		object?[] arguments, string searchDirectory)
	{
		var matchingMethod = FindNativeMethod(typeName, methodName, arguments.Length,
			searchDirectory);
		return matchingMethod == null
			? throw new NativeMethodNotFound(typeName, methodName, searchDirectory)
			: matchingMethod.IsStatic
				? matchingMethod.Invoke(null, arguments)
				: InvokeInstanceMethod(matchingMethod, arguments);
	}

	private static object? InvokeInstanceMethod(MethodInfo method, object?[] arguments)
	{
		var instance = Activator.CreateInstance(method.DeclaringType!);
		return method.Invoke(instance, arguments);
	}

	private static MethodInfo? FindNativeMethod(string typeName, string methodName,
		int argumentCount, string searchDirectory)
	{
		foreach (var dllPath in GetDllFiles(searchDirectory))
		{
			try
			{
				var assembly = Assembly.LoadFrom(dllPath);
				var matchingType = FindTypeByName(assembly, typeName);
				if (matchingType == null)
					continue;
				var matchingMethod = FindMethodByName(matchingType, methodName, argumentCount);
				if (matchingMethod != null)
					return matchingMethod;
			}
			catch (Exception)
			{
				// Skip DLLs that can't be loaded
			}
		}
		return null;
	}

	private static string[] GetDllFiles(string directory) =>
		Directory.Exists(directory)
			? Directory.GetFiles(directory, "*.dll")
			: [];

	private static System.Type? FindTypeByName(Assembly assembly, string typeName)
	{
		foreach (var type in assembly.GetExportedTypes())
			if (type.Name.Equals(typeName, StringComparison.OrdinalIgnoreCase))
				return type;
		return null;
	}

	private static MethodInfo? FindMethodByName(System.Type type, string methodName,
		int argumentCount)
	{
		foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance |
			BindingFlags.Static))
			if (method.Name.Equals(methodName, StringComparison.OrdinalIgnoreCase) &&
				method.GetParameters().Length == argumentCount)
				return method;
		return null;
	}

	public static ValueInstance ConvertToValueInstance(object? result, Language.Type returnType)
	{
		if (result == null)
			return new ValueInstance(returnType, 0);
		if (result is byte[] bytes)
			return ConvertBytesToValueInstance(bytes, returnType);
		if (result is string text)
			return new ValueInstance(text);
		if (result is double or float or int or long)
			return new ValueInstance(returnType, Convert.ToDouble(result));
		if (result is bool boolean)
			return new ValueInstance(returnType, boolean);
		return new ValueInstance(returnType, Convert.ToDouble(result));
	}

	private static ValueInstance ConvertBytesToValueInstance(byte[] bytes, Language.Type returnType)
	{
		var elementType = returnType is GenericTypeImplementation generic
			? generic.ImplementationTypes[0]
			: returnType;
		var elements = new ValueInstance[bytes.Length];
		for (var byteIndex = 0; byteIndex < bytes.Length; byteIndex++)
			elements[byteIndex] = new ValueInstance(elementType, (double)bytes[byteIndex]);
		return new ValueInstance(returnType, elements);
	}

	public sealed class NativeMethodNotFound(string typeName, string methodName,
		string searchDirectory) : Exception(
		$"No native implementation found for {typeName}.{methodName} in DLLs in {searchDirectory}");
}
