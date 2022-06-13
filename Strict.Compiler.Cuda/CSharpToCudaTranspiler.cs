using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Strict.Language;
using Strict.Language.Expressions;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Cuda;

public class CSharpToCudaTranspiler
{
	public CSharpToCudaTranspiler(Package strictPackage)
	{
		package = new Package(strictPackage, nameof(CSharpToCudaTranspiler));
		parser = new CSharpType.CSharpExpressionParser();
	}

	private readonly Package package;
	private readonly CSharpType.CSharpExpressionParser parser;

	public string Convert(string filePath)
	{
		var type = ParseCSharp(filePath);
		var resultCuda = GenerateCuda(type);
		return resultCuda;
	}

	public string GenerateCuda(Type type)
	{
		// ReSharper disable once TooManyChainedReferences
		var expression = type.Methods[0].Body.Expressions[0].ToString();
		var @operator = expression.Contains('+')
			? "+"
			: expression.Contains('-')
				? "-"
				: expression.Contains('*')
					? "*"
					: "";
		return
			@"extern ""C"" __global__ void AddNumbers(const int *first, const int *second, int* output, const int count)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * blockDim.x + ix;
	output[idx] = first[idx] " + @operator + @" second[idx];
}";
	}

	public Type ParseCSharp(string filePath) =>
		filePath == ""
			? throw new InvalidCode()
			: new CSharpType(package, filePath, parser);

	public class InvalidCode : Exception { }
}

// ReSharper disable once HollowTypeName
public class CSharpType : Type
{
	public class CSharpExpressionParser : MethodExpressionParser
	{
		public override Expression ParseAssignmentExpression(Type type, string initializationLine,
			int fileLineNumber) =>
			null!;

		public override Expression ParseMethodBody(Method method)
		{
			var binaryOperator = method.bodyLines.Last().Text.Contains('+')
				? "+"
				: method.bodyLines.Last().Text.Contains('-')
					? "-"
					: "*";
			var numberType = method.FindType(Base.Number)!;
			var leftExpression = new Value(numberType, "first");
			var rightExpression = new Value(numberType, "second");
			var returnExpression = new Return(new Binary(leftExpression,
				numberType.Methods.First(m => m.Name == binaryOperator), rightExpression));
			return new MethodBody(method, new List<Expression> { returnExpression });
		}

		public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
			null!;

		public override Expression
			TryParseExpression(Method.Line line, string remainingPartToParse) =>
			null!;
	}

	public CSharpType(Package strictPackage, string filePath, MethodExpressionParser parser) : base(
		strictPackage, filePath, parser)
	{
		var inputCode = File.ReadAllLines(filePath);
		var returnStatement = "";
		foreach (var line in inputCode)
			if (line.StartsWith("\t\t", StringComparison.Ordinal))
				returnStatement = line.Trim().Replace(";", "");
		if (returnStatement == "")
			throw new MissingReturnStatement();
		var method = new Method(this, 0, new CSharpExpressionParser(),
			new[] { "Add(first Number, second Number) returns Number", "\t" + returnStatement });
		methods.Add(method);
	}
}

public sealed class MissingReturnStatement : Exception { }

//	private bool isAtEnd()
//	{
//		return false;
//	}

//	private List<Argument> GetArguments(int startIndex, string line)
//	{
//		var endPosition = GetMethodEndPosition(startIndex, line);
//	}

//	private int GetMethodEndPosition(int startIndex, string line)
//	{
//		for (var index = startIndex; index < line.Length; index++)
//		{
//			if(line[index] == ')'))
//			return index;
//		}
//	}
//}

//internal record Argument(TokenType tokenType, string name);

//internal enum TokenType
//{
//	LeftParenthesis,
//	RightParenthesis,
//	Plus,
//	Minus,
//	Multiply,
//	Slash,
//	String,
//	Int
//}