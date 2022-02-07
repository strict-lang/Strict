using Strict.Language;
using Strict.Language.Expressions;

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
	protected override string Visit(MethodBody methodBody) => null!;

	protected override string Visit(Assignment assignment) =>
		"var " + assignment.Name + " = " + assignment.Value + ";";

	protected override string VisitWhenExpressionIsSameInCSharp(Expression expression) =>
		expression.ToString();
	/*all the same
	protected override string Visit(Binary binary) => binary.ToString();
	protected override string Visit(Boolean boolean) => boolean.ToString();
	protected override string Visit(MemberCall memberCall) => memberCall.ToString();
	protected override string Visit(MethodCall methodCall) => methodCall.ToString();
	protected override string Visit(Number number) => null!;
	protected override string Visit(Text text) => null!;
	protected override string Visit(Value value) => null!;
	*/
}