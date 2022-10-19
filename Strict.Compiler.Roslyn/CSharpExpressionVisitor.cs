using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Language;
using Strict.Language.Expressions;
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
			method.Type.Implements.Any(t => t.Name == Base.App) && method.Name == "Run";
		var methodName = isMainEntryPoint
			? "Main"
			: method.Name;
		var methodHeader =
			$"{GetAccessModifier(isInterface, method, isMainEntryPoint)}{GetCSharpTypeName(method.ReturnType)} {methodName}({WriteParameters(method)})";
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
			return $"List<{GetCSharpTypeName(((GenericType)type).ImplementationTypes[0])}>";
		return type.Name switch
		{
			Base.None => "void",
			Base.Number => "int",
			Base.Text => "string",
			Base.Boolean => "bool",
			"File" => "FileStream",
			_ => type.Name
		};
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

	protected override string Visit(Assignment assignment) =>
		"var " + assignment.Name + " = " + Visit(assignment.Value);

	protected override string Visit(Return returnExpression) =>
		"return " + Visit(returnExpression.Value);

	protected override string Visit(MethodCall methodCall) =>
		VisitMethodCallInstance(methodCall) +
		(methodCall.Method.Name == Method.From || methodCall.Instance == null
			? ""
			: "." + (methodCall.Method.Name == "Read" && methodCall.Instance.ToString() == "file"
				? "ReadToEnd"
				: methodCall.Method.Name == "Write" &&
				methodCall.Instance.ReturnType.Name is Base.Log or Base.System
					? "WriteLine"
					: methodCall.Method.Name)) + "(" + methodCall.Arguments.Select(Visit).ToWordList() +
		(methodCall.ReturnType.Name == "File"
			? ", FileMode.OpenOrCreate"
			: "") + ")";

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
		memberCall.Member.Type.Name is Base.Log or Base.System
			? "Console"
			: memberCall.Instance != null
				? memberCall.Instance + "." + memberCall.Member.Name
				: memberCall.Member.Name;

	protected override string Visit(Value value) =>
		value.Data is Type
			? GetCSharpTypeName(value.ReturnType)
			: value.ToString();
}