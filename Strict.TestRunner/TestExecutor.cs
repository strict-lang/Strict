using Strict.HighLevelRuntime;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.TestRunner;

/// <summary>
/// All methods have to have tests, and those are executed automatically after parsing. However, if
/// we don't call some code, it is not parsed, executed, or tested at all. This forces execution
/// of every method in every type to run all included tests to find out if anything is not working.
/// </summary>
public sealed class TestExecutor
{
	private readonly Executor executor = new(TestBehavior.TestRunner);

	public void RunAllTestsInPackage(Package package)
	{
		foreach (var type in package)
			RunAllTestsInType(type);
	}

	public void RunAllTestsInType(Type type)
	{
		foreach (var method in type.Methods)
			if (!method.IsTrait)
				RunMethod(method);
	}

	public void RunMethod(Method method) => executor.Execute(method, null, []);
}