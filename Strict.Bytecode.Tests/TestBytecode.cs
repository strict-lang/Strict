using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public class TestBytecode : TestExpressions
{
	protected static readonly Type NumberType = TestPackage.Instance.GetType(Type.Number);
	protected static readonly Type ListType = TestPackage.Instance.GetType(Type.List);
	protected static ValueInstance Number(double value) => new(NumberType, value);
	protected static ValueInstance Text(string value) => new(value);

	protected MethodCall GenerateMethodCallFromSource(string programName, string methodCall,
		params string[] source)
	{
		if (type.Package.FindDirectType(programName) == null)
			new Type(type.Package, new TypeLines(programName, source)).ParseMembersAndMethods(
				new MethodExpressionParser());
		return (MethodCall)ParseExpression(methodCall);
	}
}