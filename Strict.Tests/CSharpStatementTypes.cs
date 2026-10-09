using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Tests;

/// <summary>
/// "TypeName:line Type" of every method line the C# parser typed, in the format of the Strict
/// Expressions/TypeReport.strict program: lists use plural names, mutable types their inner type,
/// other generics only their generic name (dictionary key and value types are not inferred yet).
/// </summary>
public static class CSharpStatementTypes
{
	public static async Task<HashSet<string>> Collect(string root, string folder)
	{
		var repositories = new Repositories(new MethodExpressionParser());
		var package = folder == "."
			? await repositories.LoadStrictPackage()
			: await repositories.LoadStrictPackage(nameof(Strict) + Context.ParentSeparator + folder);
		var folderPath = Path.GetFullPath(Path.Combine(root, folder));
		var lines = new HashSet<string>();
		foreach (var type in package.Types.Values.ToList())
			if (type is not GenericTypeImplementation && File.Exists(type.FilePath) &&
				Path.GetDirectoryName(type.FilePath) == folderPath)
				foreach (var method in type.Methods.Where(method => !method.IsTrait))
				{
					var body = method.GetBodyAndParseIfNeeded();
					foreach (var test in method.Tests)
						Add(lines, method, test);
					Add(lines, method, body);
				}
		return lines;
	}

	private static void Add(HashSet<string> lines, Method method, Expression expression)
	{
		switch (expression)
		{
		case Body body:
			foreach (var child in body.Expressions)
				Add(lines, method, child);
			return;
		case If ifExpression when ifExpression.Then is not Body &&
			ifExpression.Then.LineNumber == ifExpression.LineNumber:
			AddLine(lines, method, ifExpression.LineNumber, ifExpression.ReturnType);
			return;
		case If ifExpression:
			Add(lines, method, ifExpression.Then);
			if (ifExpression.OptionalElse != null)
				Add(lines, method, ifExpression.OptionalElse);
			return;
		case For forExpression:
			AddLine(lines, method, forExpression.LineNumber, forExpression.Iterator.ReturnType);
			Add(lines, method, forExpression.Body);
			return;
		case Declaration declaration:
			AddLine(lines, method, declaration.LineNumber, declaration.Value.ReturnType);
			return;
		case MutableReassignment reassignment:
			AddLine(lines, method, reassignment.LineNumber, reassignment.Value.ReturnType);
			return;
		case Return returnExpression:
			AddLine(lines, method, returnExpression.LineNumber, returnExpression.Value.ReturnType);
			return;
		default:
			AddLine(lines, method, expression.LineNumber, expression.ReturnType);
			return;
		}
	}

	private static void AddLine(HashSet<string> lines, Method method, int lineNumber, Type type)
	{
		if (lineNumber > method.TypeLineNumber)
			lines.Add(method.Type.Name + ":" + (lineNumber + 1) + " " + StrictName(type));
	}

	private static string StrictName(Type type) =>
		type.IsError
			? Type.Error
			: type is GenericTypeImplementation generic
			? generic.Generic.IsMutable
				? StrictName(generic.ImplementationTypes[0])
				: generic.Generic.IsList
					? StrictName(generic.ImplementationTypes[0]) + "s"
					: generic.Generic.Name
				: type is GenericType genericType
					? genericType.Generic.Name
					: type.Name;
}