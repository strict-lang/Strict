using System.IO;
using Strict.Language;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Cuda;

public class CSharpToCudaTranspiler(Package strictBase) : IDisposable
{
	private readonly Package package = new(strictBase, nameof(CSharpToCudaTranspiler));
	private readonly CSharpType.CSharpExpressionParser parser = new();

	public string Convert(string filePath)
	{
		var type = ParseCSharp(filePath);
		return GenerateCuda(type);
	}

	public static string GenerateCuda(Type type) =>
		new StatementsToCudaCompiler().Compile(type.Methods[0]);

	public Type ParseCSharp(string filePath) =>
		filePath == ""
			? throw new InvalidCode()
			: new CSharpType(package, filePath);

	public class InvalidCode : Exception;

	public void Dispose() => package.Dispose();
}

public class CSharpType : Type
{
	public class CSharpExpressionParser : MethodExpressionParser
	{
		/*this is a hack anyway
		public override BlockExpression ParseMethodBody(Method method)
		{
			if (method.bodyLines.Last().Text.Contains("depth"))
				return new MethodBody(method,
					new List<Expression> { new Text(method, method.bodyLines.Last().Text) });
			var binaryOperator = method.bodyLines.Last().Text.Contains('+')
				? "+"
				: method.bodyLines.Last().Text.Contains('-')
					? "-"
					: "*";
			var numberType = method.FindType(Type.Number)!;
			var arguments = new Expression[] { new Value(numberType, "second") };
			var returnExpression = new Return(new Binary(new Value(numberType, "first"),
				numberType.GetMethod(binaryOperator, arguments), arguments));
			return new MethodBody(method, new List<Expression> { returnExpression });
		}
		*/
	}

	public CSharpType(Package strictPackage, string filePath)
		: this(strictPackage, ExtractMethodInfo(filePath)) { }

	private CSharpType(Package strictPackage, (string typeName, string methodLine, string bodyLine) info)
		: base(strictPackage, new TypeLines(info.typeName, info.methodLine))
	{
		methods.Add(new Method(this, 0, new CSharpExpressionParser(),
			[info.methodLine, "\t" + info.bodyLine]));
	}

	private static (string typeName, string methodLine, string bodyLine) ExtractMethodInfo(string filePath)
	{
		var typeName = Path.GetFileNameWithoutExtension(filePath);
		var methodName = "";
		var returnType = "";
		var parameters = new List<string>();
		var returnStatement = "";
		foreach (var line in global::System.IO.File.ReadAllLines(filePath))
		{
			if (IsIgnoredOrEmptyText(line, typeName))
				continue;
			if (line.StartsWith("\t\treturn", StringComparison.Ordinal) || line.StartsWith("\t\t\t", StringComparison.Ordinal))
			{
				var value = line.Trim().Replace(";", "");
				if (value.StartsWith("return ", StringComparison.Ordinal))
					value = value[7..];
				else if (value.Contains(" = "))
					value = value[(value.LastIndexOf(" = ", StringComparison.Ordinal) + 3)..];
				returnStatement = value;
			}
			else
			{
				var parts = line.Trim().Split([' ', '(', ')', ','], StringSplitOptions.RemoveEmptyEntries);
				if (parts[1] == "float")
					returnType = " Number";
				methodName = parts[2];
				AddMethodParameters(parts, parameters);
			}
		}
		if (returnStatement == "")
			throw new MissingReturnStatement();
		return (typeName, methodName + parameters.ToBrackets() + returnType, returnStatement);
	}

	private static bool IsIgnoredOrEmptyText(string line, string typeName) =>
		line == "" || line.Contains("{") || line.Contains("}") ||
		line.StartsWith("using ", StringComparison.Ordinal) ||
		line.StartsWith("namespace ", StringComparison.Ordinal) ||
		line.Contains(typeName) || line.Contains("this.") ||
		line.StartsWith("\tprivate ", StringComparison.Ordinal) ||
		line.StartsWith("\t\tfor ", StringComparison.Ordinal);

	private static void AddMethodParameters(IReadOnlyList<string> parts, List<string> parameters)
	{
		for (var index = 3; index < parts.Count; index += 2)
			if (parts[index] == "DepthImage")
				parameters.AddRange([
					"input Number", "Width Number", "Height Number", "initialDepth Number"
				]);
			else if (parts[index] != "float")
				throw new NotSupportedException(parts[index + 1]);
			else
				parameters.Add(parts[index + 1] + " Number");
	}
}