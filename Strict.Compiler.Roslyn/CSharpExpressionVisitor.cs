using Strict.Language;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public class CSharpExpressionVisitor : ExpressionVisitor
{
	protected override IReadOnlyList<string> VisitBody(Body methodBody) =>
		Indent(methodBody.Expressions.Select(VisitBody)).ToList();

	private static IEnumerable<string> Indent(IEnumerable<IReadOnlyList<string>> expressions) =>
		Indent(expressions.SelectMany(line => line));

	public string VisitMethodHeader(Method method, bool isInterface)
	{
		var isMainEntryPoint =
			method.Type.Members.Any(t => t.Type.Name == Base.App) && method.Name == "Run";
		var methodName = isMainEntryPoint
			? "Main"
			: method.Name;
		var methodHeader =
			$"{
				GetAccessModifier(isInterface, method, isMainEntryPoint)
			}{
				GetCSharpTypeName(method.ReturnType)
			} {
				methodName
			}({
				WriteParameters(method)
			})";
		return isInterface
			? methodHeader + ";" + NewLine
			: methodHeader + NewLine + "\t{" + NewLine + "\t";
	}

	public string GetAccessModifier(bool isTrait, Method method, bool isMainEntryPoint) =>
		isTrait
			? ""
			: (method.IsPublic
				? "public "
				: "private ") + (isMainEntryPoint
				? "static "
				: "");

	public string GetCSharpTypeName(Type type)
	{
		if (type.IsList)
			return $"List<{GetCSharpTypeName(type.GetFirstImplementation())}>";
		if (type.IsNone)
			return "void";
		if (type.IsBoolean)
			return "bool";
		if (type.IsNumber)
			return "int"; //could be double as well
		if (type.IsText)
			return "string";
		return type.Name == "File"
			? "FileStream"
			: type.Name;
	}

	private string WriteParameters(Method method) =>
		string.Join(", ",
			method.Parameters.Select(p => GetCSharpTypeName(p.Type) + " " + p.Name));

	private static readonly string NewLine = Environment.NewLine;

	protected override string GetBinaryOperator(string methodName) =>
		methodName switch
		{
			BinaryOperator.Is => "==",
			_ => methodName
		};

	protected override string Visit(Declaration declaration) =>
		"var " + declaration.Name + " = " + Visit(declaration.Value);

	protected override string Visit(Return returnExpression) =>
		"return " + Visit(returnExpression.Value);

	protected override string Visit(MethodCall methodCall)
	{
		var result = VisitMethodCallInstance(methodCall);
		if (methodCall.Method.Name != Method.From && methodCall.Instance != null)
			result += ".";
		if (methodCall.Method.Name == "Read" && methodCall.Instance?.ToString() == "file")
			result += "ReadToEnd"; //ncrunch: no coverage
		else if (methodCall.Method.Name is "Write" or "Log" &&
			methodCall.Instance?.ReturnType.Name is Base.Logger or Base.System)
			result += "WriteLine";
		else if (methodCall.Method.Name == Method.From)
			result += methodCall.Method.Type.Name == "File"
				? "FileStream"
				: methodCall.Method.Type.Name;
		else
			result += methodCall.Method.Name;
		result += "(" + methodCall.Arguments.Select(Visit).ToWordList();
		if (methodCall.ReturnType.Name == "File")
			result += ", FileMode.OpenOrCreate";
		result += ")";
		return result;
	}

	private string VisitMethodCallInstance(MethodCall methodCall) =>
		((methodCall.Arguments.FirstOrDefault() as MethodCall)?.Instance?.ToString() == "file"
			? "using var reader = new StreamReader(file);"
			: "") + (methodCall.Method.Name == Method.From
			? "new "
			: "") + (methodCall.Instance != null
			? methodCall.Instance.ToString() == "file" && methodCall.Method.Name == "Write"
				? "using var writer = new StreamWriter(file); writer"
				: methodCall.Instance.ToString() == "file" && methodCall.Method.Name == "Read"
					? "reader"
					: Visit(methodCall.Instance)
			: "");

	protected override string Visit(MemberCall memberCall) =>
		memberCall.Member.Type.Name is Base.Logger or Base.System
			? "Console"
			: memberCall.Instance != null
				? memberCall.Instance + "." + memberCall.Member.Name
				: memberCall.Member.Name;

	protected override string Visit(Value value) =>
		value.Data.IsValueTypeInstanceType
			? GetCSharpTypeName(value.ReturnType)
			: value.ToString();

	protected override IReadOnlyList<string> VisitFor(For forExpression)
	{
		var block = new List<string> { "foreach (var index in " + Visit(forExpression.Iterator) + ")" };
		block.AddRange(Indent(VisitBody(forExpression.Body)));
		return block;
	}
}