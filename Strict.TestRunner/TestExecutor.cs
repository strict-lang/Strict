using Strict.HighLevelRuntime;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.TestRunner;

/// <summary>
/// All methods have to have tests, and those are executed automatically after parsing. However, if
/// we don't call some code, it is not parsed, executed, or tested at all. This forces execution
/// of every method in every type to run all included tests to find out if anything is not working.
/// </summary>
public sealed class TestExecutor(Package package) : Executor(package, TestBehavior.TestRunner)
{
	private readonly Dictionary<Package, Type[]> cachedNonGenericTypes = new();

	public void RunAllTestsInPackage(Package package)
	{
		Statistics.PackagesTested++;
		if (!cachedNonGenericTypes.TryGetValue(package, out var types))
			cachedNonGenericTypes[package] = types = BuildNonGenericTypesArray(package);
		foreach (var type in types)
			RunAllTestsInType(type);
	}

	private static Type[] BuildNonGenericTypesArray(Package package)
	{
		var result = new List<Type>(package.Types.Count);
		foreach (var type in package.Types.Values)
			if (type is not GenericTypeImplementation)
				result.Add(type);
		return result.ToArray();
	}

	public void RunAllTestsInType(Type type)
	{
		Statistics.TypesTested++;
		foreach (var method in type.Methods)
			if (!method.IsTrait)
				RunMethod(method);
	}

	public void RunMethod(Method method)
	{
		Statistics.MethodsTested++;
		Execute(method);
	}
}