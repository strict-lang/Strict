using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public sealed class CSharpTypeVisitor : TypeVisitor
{
	public CSharpTypeVisitor(Type type)
	{
		Name = type.Name;
		expressionVisitor = new CSharpExpressionVisitor();
		isImplementingApp = type.Members.Any(t => t.Type.Name == Base.App);
		isInterface = type.IsTrait;
		CreateHeader(type);
		CreateClass();
		foreach (var member in type.Members)
			VisitMember(member);
		foreach (var method in type.Methods)
			VisitMethod(method);
		AddTests();
		ParsingDone();
	}

	public string Name { get; }
	private readonly CSharpExpressionVisitor expressionVisitor;
	private readonly bool isImplementingApp;
	private readonly bool isInterface;

	private void CreateHeader(Type type)
	{
		foreach (var member in type.Members)
			if (type.IsTraitImplementation(member.Type))
				VisitImplement(member.Type);
		FileContent += "namespace " + type.Package.FolderPath + SemicolonAndLineBreak + NewLine;
	}

	public string FileContent { get; private set; } = "";

	public void VisitImplement(Type type)
	{
		if (isImplementingApp)
			return;
		baseClasses += (baseClasses.Length > 0
			? ", "
			: " : ") + type.Name;
	}

	private string baseClasses = "";

	private void CreateClass() =>
		FileContent += "public " + (isInterface
			? "interface"
			: "class") + " " + Name + baseClasses + NewLine + "{" + NewLine;

	private static readonly string NewLine = Environment.NewLine;

	public void VisitMember(Member member)
	{
		if (member.Name == "logger" || member.Name == "App")
			return;
		var accessModifier = member.IsPublic
			? "public"
			: "private";
		var csharpTypeName = expressionVisitor.GetCSharpTypeName(member.Type);
		var initializationExpression =
			BuildInitializationExpression(member, csharpTypeName, ref accessModifier);
		FileContent += "\t" + accessModifier + " " + csharpTypeName + " " +
			member.Name + initializationExpression + SemicolonAndLineBreak;
	}

	private string BuildInitializationExpression(Member member, string csharpTypeName,
		ref string accessModifier)
	{
		var initializationExpression = "";
		if (member.InitialValue != null)
			initializationExpression += " = " + expressionVisitor.Visit(member.InitialValue);
		if (member.Name == "file")
			accessModifier += " static";
		if (string.IsNullOrEmpty(initializationExpression) && member.Type.IsIterator)
			initializationExpression += $" = new {csharpTypeName}()";
		return initializationExpression;
	}

	private static readonly string SemicolonAndLineBreak = ";" + NewLine;

	public void VisitMethod(Method method)
	{
		VisitMethodHeader(method);
		if (!isInterface)
			VisitMethodBody(method);
	}

	private void VisitMethodHeader(Method method) => FileContent += "\t" + expressionVisitor.VisitMethodHeader(method, isInterface);

	private void VisitMethodBody(Method method)
	{
		var body = expressionVisitor.VisitBody(method.GetBodyAndParseIfNeeded());
		testExpressions.Add(method.Name,
			body.Where(line =>
				line.StartsWith("\tnew ", StringComparison.Ordinal) && line.Contains("==")));
		FileContent += $"{
			(body.Count == 1
				? "\t" + body[0]
				: string.Join(LineBreakAndSpace,
					body.Where(line =>
						!line.StartsWith("\tnew ", StringComparison.Ordinal) || !line.Contains("=="))))
		}{
			LineBreakAndSpace
		}}}{
			NewLine
		}";
	}

	private readonly Dictionary<string, IEnumerable<string>> testExpressions = new();
	private static readonly string LineBreakAndSpace = NewLine + "\t";

	private void AddTests()
	{
		var hasTestMethods = false;
		foreach (var testMethod in testExpressions)
		{
			if (!testMethod.Value.Any())
				break;
			hasTestMethods = true;
			AddTestExpressions(testMethod);
		}
		if (hasTestMethods)
			FileContent += $"{NewLine}\t}}{NewLine}";
	}

	private void AddTestExpressions(KeyValuePair<string, IEnumerable<string>> testMethod)
	{
		FileContent += $"{NewLine}\t[Test]" + $"{NewLine}\tpublic void {testMethod.Key}Test()" +
			$"{NewLine}\t{{";
		foreach (var test in testMethod.Value)
			FileContent += $"{NewLine}\t\tAssert.That(() => {test[1..^1]}));";
	}

	public void ParsingDone() => FileContent += "}";
}