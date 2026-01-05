using Strict.HighLevelRuntime;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.TestRunner;

public sealed class TestExecutor(Package basePackage)
{
	private readonly Executor executor = new(basePackage, true);

	public bool RunMethod(Method method)
	{
		try
		{
			executor.Execute(method, null, []);
			return true;
		}
		catch (Exception)
		{
			return false;
		}
	}

	public bool RunTests(Method method)
	{
		try
		{
			executor.Execute(method, null, []);
			return true;
		}
		catch (Exception)
		{
			return false;
		}
	}

	public bool RunTests(Type type) => type.Methods.All(RunTests);
}