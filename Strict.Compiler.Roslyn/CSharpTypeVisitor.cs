using System;
using System.Linq;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Roslyn;

public class CSharpTypeVisitor : TypeVisitor
{
	public CSharpTypeVisitor(Type type)
	{
		Name = type.Name;
		isImplementingApp = type.Implements.Any(t => t.Name == Base.App);
		isInterface = type.IsTrait;
		foreach (var import in type.Imports)
			VisitImport(import);
		foreach (var implement in type.Implements)
			VisitImplement(implement);
		FileContent += "namespace " + type.Package.FolderPath + SemicolonAndLineBreak + NewLine;
		CreateClass();
		foreach (var member in type.Members)
			VisitMember(member);
		foreach (var method in type.Methods)
			VisitMethod(method);
		ParsingDone();
	}

	public string Name { get; }
	private readonly bool isImplementingApp;
	private readonly bool isInterface;
	public string FileContent { get; private set; }

	public void VisitImport(Package import) =>
		FileContent += "using " + import.Name + SemicolonAndLineBreak;

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
		var initializationExpression = "";
		if (member.Value != null)
			initializationExpression += " = " + new CSharpExpressionVisitor(member.Value).Visit();
		FileContent += "\t" + accessModifier + " " + GetCSharpTypeName(member.Type) + " " +
			member.Name + initializationExpression + SemicolonAndLineBreak;
	}

	private static readonly string SemicolonAndLineBreak = ";" + NewLine;

	public void VisitMethod(Method method)
	{
		var accessModifier = isInterface
			? ""
			: method.IsPublic
				? "public "
				: "private ";
		var isMainEntryPoint = isImplementingApp && method.Name == "Run";
		var staticMain = isMainEntryPoint
			? "static "
			: "";
		var methodName = isMainEntryPoint
			? "Main"
			: method.Name;
		FileContent +=
			$"\t{accessModifier}{staticMain}{GetCSharpTypeName(method.ReturnType)} " +
			$"{methodName}({WriteParameters(method)})";
		FileContent += isInterface
			? ";" + NewLine
			: WriteMethodBody(method);
	}

	private static string WriteMethodBody(Method method) =>
		$"{NewLine}\t{{{NewLine}" + new CSharpExpressionVisitor(method).Visit() + "\t}" + NewLine;

	private static string WriteParameters(Method method) =>
		string.Join(", ",
			method.Parameters.Select(p => GetCSharpTypeName(p.Type) + " " + p.Name));

	private static string GetCSharpTypeName(Context type) =>
		type.Name switch
		{
			Base.None => "void",
			Base.Number => "int",
			"File" => "FileStream",
			_ => type.Name //ncrunch: no coverage
		};

	public void ParsingDone() => FileContent += "}";
}