using System;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public class CSharpGenerator : SourceGenerator
{
	public SourceFile Generate(Type type) => new CSharpFile(type);
}

//ncrunch: no coverage start, TODO later
public interface ExpressionVisitor { }

public class CSharpExpressionVisitor : ExpressionVisitor
{
	// ReSharper disable once UnusedParameter.Local
	public CSharpExpressionVisitor(Method method) { }

	public string Visit(int tabIndentation = 2) =>
		new string('\t', tabIndentation) + "Console.WriteLine(\"Hello World\");" +
		Environment.NewLine;
}
