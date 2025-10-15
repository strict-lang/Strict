using Strict.HighLevelRuntime;
using Strict.Language;

namespace Strict.TestRunner;

public sealed class TestExecutor(Package basePackage)
{
	private readonly Executor executor = new(basePackage);

	public bool RunMethod(Method method) =>
		executor.Execute(method, null, []) is { ReturnType.Name: Base.Boolean, Value: true };
}