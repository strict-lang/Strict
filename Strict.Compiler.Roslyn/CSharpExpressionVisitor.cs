using System;
using Strict.Language;

namespace Strict.Compiler.Roslyn;

public class CSharpExpressionVisitor : ExpressionVisitor
{
	// ReSharper disable once UnusedParameter.Local
	public CSharpExpressionVisitor(Method method) =>
		expression = new MethodBody(method, Array.Empty<Expression>()); //method.Body;

	public CSharpExpressionVisitor(Expression expression) => this.expression = expression;
	private readonly Expression expression;

	public string Visit(int tabIndentation = 2) =>
		expression.ReturnType.Name == "File"
			? "new FileStream(\"test.txt\", FileMode.Open)"
			: new string('\t', tabIndentation) + "Console.WriteLine(\"Hello World\");" +
			Environment.NewLine;
}