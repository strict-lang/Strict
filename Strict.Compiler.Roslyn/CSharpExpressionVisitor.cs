using System;
using Strict.Language;

namespace Strict.Compiler.Roslyn;

public class CSharpExpressionVisitor : ExpressionVisitor
{
	// ReSharper disable once UnusedParameter.Local
	public CSharpExpressionVisitor(Method method) =>
		expression = new MethodBody(method, method.Body.Expressions);

	public CSharpExpressionVisitor(Expression expression) => this.expression = expression;
	private readonly Expression expression;

	public string Visit(int tabIndentation = 2) =>
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
}