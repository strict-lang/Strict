using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public class CSharpTypeVisitor : TypeVisitor
{
	public CSharpTypeVisitor(Type type)
	{
		Name = type.Name;
		expressionVisitor = new CSharpExpressionVisitor();
		isImplementingApp = type.Implements.Any(t => t.Name == Base.App);
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
		foreach (var implement in type.Implements)
			VisitImplement(implement);
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
		if (member.Name == "log")
			return;
		var accessModifier = member.IsPublic
			? "public"
			: "private";
		var csharpTypeName = expressionVisitor.GetCSharpTypeName(member.Type);
		var initializationExpression = BuildInitializationExpression(member, csharpTypeName, ref accessModifier);
		FileContent += "\t" + accessModifier + " " + csharpTypeName + " " +
			member.Name + initializationExpression + SemicolonAndLineBreak;
	}

	private string BuildInitializationExpression(Member member, string csharpTypeName,
		ref string accessModifier)
	{
		var initializationExpression = "";
		if (member.Value != null)
			initializationExpression += " = " + expressionVisitor.Visit(member.Value);
		if (member.Name == "file")
			accessModifier += " static";
		if (string.IsNullOrEmpty(initializationExpression) && member.Type.IsList)
			initializationExpression += $" = new {csharpTypeName}()";
		return initializationExpression;
	}

	private static readonly string SemicolonAndLineBreak = ";" + NewLine;

	public void VisitMethod(Method method)
	{
		var body = expressionVisitor.VisitBody(isInterface
			? new Body(method)
			: method.GetBodyAndParseIfNeeded());
		testExpressions.Add(method.Name,
			body.Where(line =>
				line.StartsWith("\tnew ", StringComparison.Ordinal) && line.Contains("==")));
		FileContent += "\t" + string.Join(NewLine + "\t",
			body.Where(line =>
				!line.StartsWith("\tnew ", StringComparison.Ordinal) || !line.Contains("=="))) + NewLine;
	}

	private Dictionary<string,IEnumerable<string>> testExpressions = new();
	public void ParsingDone() => FileContent += "}";

	private void AddTests()
	{
		foreach (var testMethod in testExpressions)
		{
			FileContent += $"{NewLine}\t[Test]" + $"{NewLine}\tpublic void {testMethod.Key}()" + $"{NewLine}\t{{";
			foreach (var test in testMethod.Value)
				FileContent += $"{NewLine}\t\tAssert.That(() => {test[1..^1]}));";
		}
		FileContent += $"{NewLine}\t}}{NewLine}";
	}
}