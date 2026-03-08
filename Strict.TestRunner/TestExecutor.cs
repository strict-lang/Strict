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
	public void RunAllTestsInPackage(Package package)
	{
		Statistics.PackagesTested++;
		var typesToTest = new List<Type>();
		foreach (var pair in package.Types)
			if (pair.Value is not GenericTypeImplementation)
				typesToTest.Add(pair.Value);
		foreach (var type in typesToTest)
			RunAllTestsInType(type);
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