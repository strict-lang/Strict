using System.Collections.Generic;
using System.Linq;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

	/*old
		expression.ReturnType.Name == "File"
			? "new FileStream(" + ((MethodBody)expression).Expressions[0] + ", FileMode.OpenOrCreate)"
			: expression is MethodBody body && body.Expressions.Count == 3
				? @"var program = new CreateFileWriteIntoItReadItAndThenDeleteIt();
TextWriter writer = new StreamWriter(program.file);
writer.Write(""Hello"");
writer.Flush();
program.file.Seek(0, SeekOrigin.Begin);
TextReader reader = new StreamReader(program.file);
Console.Write(reader.ReadToEnd());
program.file.Close();
File.Delete(""temp.txt"");
"
				: new string('\t', tabIndentation) + "Console.WriteLine(\"Hello World\");" +
				Environment.NewLine;
	*/
public class CSharpExpressionVisitor : ExpressionVisitor
{
	protected override IReadOnlyList<string> VisitBlock(MethodBody methodBody)
	{
		var method = methodBody.Method;
		var isMainEntryPoint = method.Type.Implements.Any(t => t.Name == Base.App) && method.Name == "Run";
		var staticMain = isMainEntryPoint
			? "static "
			: "";
		var methodName = isMainEntryPoint
			? "Main"
			: method.Name;
		var isInterface = methodBody.Method.Type.IsTrait || methodBody.Expressions.Count == 0;
		var methodHeader =
			$"{GetAccessModifier(isInterface, method)}{staticMain}{GetCSharpTypeName(method.ReturnType)} {methodName}({WriteParameters(method)})";
		if (isInterface)
			return new[] { methodHeader + ";" };
		var methodLines = new List<string> { methodHeader, "{" };
		methodLines.AddRange(Indent(methodBody.Expressions.Select(VisitBlock)));
		methodLines.Add("}");
		return methodLines;
	}

	private string WriteParameters(Method method) =>
		string.Join(", ",
			method.Parameters.Select(p => GetCSharpTypeName(p.Type) + " " + p.Name));

	private static IEnumerable<string> Indent(IEnumerable<IReadOnlyList<string>> expressions) =>
		Indent(expressions.SelectMany(line => line));

	public string GetAccessModifier(bool isTrait, Method method) =>
		isTrait
			? ""
			: method.IsPublic
				? "public "
				: "private ";

	public string GetCSharpTypeName(Type type) =>
		type.Name switch
		{
			Base.None => "void",
			Base.Number => "int",
			Base.Boolean => "bool",
			"File" => "FileStream",
			_ => type.Name
		};

	protected override string GetBinaryOperator(string methodName) =>
		methodName switch
		{
			BinaryOperator.Is => "==",
			_ => methodName
		};

	protected override string Visit(Assignment assignment) =>
		"var " + assignment.Name + " = " + Visit(assignment.Value);

	protected override string Visit(Return returnExpression) =>
		"return " + Visit(returnExpression.Value);

	protected override string Visit(MethodCall methodCall) =>
		(methodCall.Method.Name == Method.From
			? "new "
			: "") + Visit(methodCall.Instance) + (methodCall.Method.Name == Method.From
			? ""
			: "." + (methodCall.Method.Name == "Write" && methodCall.Instance.ReturnType ==
				methodCall.ReturnType.GetType(Base.Log) || methodCall.Instance.ReturnType ==
				methodCall.ReturnType.FindType(Base.System)
					? "WriteLine"
					: methodCall.Method.Name)) + "(" +
		string.Join(", ", methodCall.Arguments.Select(Visit)) + (methodCall.ReturnType.Name == "File"
			? ", FileMode.OpenOrCreate"
			: "") + ")";

	protected override string Visit(MemberCall memberCall) =>
		memberCall.Member.Type == memberCall.ReturnType.GetType(Base.Log) ||
		memberCall.Member.Type == memberCall.ReturnType.FindType(Base.System)
			? "Console"
			: memberCall.Parent != null
				? memberCall.Parent + "." + memberCall.Member.Name
				: memberCall.Member.Name;

	protected override string Visit(Value value) =>
		value.Data is Type
			? GetCSharpTypeName(value.ReturnType)
			: value.ToString();

	protected override string VisitWhenExpressionIsSameInCSharp(Expression expression) =>
		expression.ToString();
}