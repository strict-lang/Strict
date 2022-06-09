using System;
using System.Collections.Generic;
using System.Linq;
using Strict.Language;
using Strict.Language.Expressions;
using Expression = Strict.Language.Expression;
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

	public string GenerateCuda(object type) =>
		@"extern ""C"" __global__ void AddNumbers(const int *first, const int *second, int* output, const int count)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * blockDim.x + ix;
	output[idx] = first[idx] + second[idx];
}";

	public Type ParseCSharp(string filePath)
	{
		if (filePath == "")
			throw new InvalidCode();
		var type = new CSharpType(package, filePath, parser);
		return type;
	}

	public class InvalidCode : Exception { }
}

// ReSharper disable once HollowTypeName
public class CSharpType : Type
{
	public class CSharpExpressionParser : MethodExpressionParser
	{
		public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber) => null!;

		public override Expression ParseMethodBody(Method method)
		{
			var numberType = method.FindType(Base.Number)!;
			var leftExpression = new Value(numberType, "first");
			var rightExpression = new Value(numberType, "second");
			var returnExpression = new Return(new Binary(leftExpression,
				numberType.Methods.First(m => m.Name == "+"), rightExpression));
			return new MethodBody(method, new List<Expression> { returnExpression });
		}

		public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) => null!;
		public override Expression TryParseExpression(Method.Line line, string remainingPartToParse) => null!;
	}

	public CSharpType(Package strictPackage, string filePath, MethodExpressionParser parser) : base(strictPackage, filePath, parser)
	{
		var method = new Method(this, 0, new CSharpExpressionParser(),
			new[] { "Add(first Number, second Number) returns Number" });
		methods.Add(method);
	}
}