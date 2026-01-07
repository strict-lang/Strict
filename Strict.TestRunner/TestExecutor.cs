using Strict.HighLevelRuntime;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.TestRunner;

public sealed class TestExecutor(Package basePackage)
{
	private readonly Executor executor = new(basePackage);

	public void RunTests(Type type)
	{
		foreach (var method in type.Methods)
			try
			{
				executor.Execute(method, null, []);
			}
			catch (Executor.InlineTestFailed ex)
			{
				throw new TestFailed(method, ex);
			}
	}

	public sealed class TestFailed(Method method, Executor.InlineTestFailed ex)
		: Exception($"Test {method.Name} failed in {method.Type.Name} at line {ex.Line.LineNumber}: {ex.Message}");
}