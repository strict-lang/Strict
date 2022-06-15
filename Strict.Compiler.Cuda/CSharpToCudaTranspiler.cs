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
		return GenerateCuda(type);
	}

	// ReSharper disable once MethodTooLong
	public static string GenerateCuda(Type type)
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
		var parameterText = "";
		string output;
		foreach (var parameter in type.Methods[0].Parameters)
		{
			if (parameter.Type.Name != Base.Number)
				throw new NotSupportedException(parameter.ToString());
			if (parameter.Name is "Width" or "Height")
				parameterText += "const int " + parameter.Name + ", ";
			else if (parameter.Name == "initialDepth")
				parameterText += "const float " + parameter.Name + ", ";
			else
				parameterText += "const float *" + parameter.Name + ", ";
		}
		parameterText += "float *output";
		if (!parameterText.Contains("Width"))
		{
			parameterText += ", const int count";
			output = "first[idx] " + @operator + @" second[idx]";
		}
		else
		{
			output = "initialDepth";
		}
		return
			@"extern ""C"" __global__ void " + type.Methods[0].Name + "(" + parameterText + @")
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = y * blockDim.x + x;
	output[idx] = " + output + @";
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
		public override Expression ParseMethodBody(Method method)
		{
			if (method.bodyLines.Last().Text.Contains("depth"))
				return new MethodBody(method,
					new List<Expression> { new Text(method, method.bodyLines.Last().Text) });
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

		//Method not implemented. Added dummy code to avoid Teamcity ForcedNull error
		public override Expression ParseAssignmentExpression(Type type, string initializationLine,
			int fileLineNumber) =>
			type.Methods[0].Body;

		//Method not implemented. Added dummy code to avoid Teamcity ForcedNull error
		public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
			line.Method.Body;

		//Method not implemented. Added dummy code to avoid Teamcity ForcedNull error
		public override Expression
			TryParseExpression(Method.Line line, string remainingPartToParse) =>
			line.Method.Body;
	}

	// ReSharper disable once CyclomaticComplexity
	public CSharpType(Package strictPackage, string filePath, MethodExpressionParser parser) : base(
		strictPackage, filePath, parser)
	{
		var inputCode = File.ReadAllLines(filePath);
		var methodName = "";
		var returnType = "";
		var parameters = new List<string>();
		var returnStatement = "";
		foreach (var line in inputCode)
		{
			// ReSharper disable once ComplexConditionExpression
			if (line == "" || line.Contains("{") || line.Contains("}") || line.StartsWith("using ", StringComparison.Ordinal) || line.StartsWith("namespace ", StringComparison.Ordinal) ||
				line.Contains(Name) || line.Contains("this.") || line.StartsWith("\tprivate ", StringComparison.Ordinal) || line.StartsWith("\t\tfor ", StringComparison.Ordinal))
				continue;
			if (line.StartsWith("\t\treturn", StringComparison.Ordinal) || line.StartsWith("\t\t\t", StringComparison.Ordinal))
				returnStatement = line.Trim().Replace(";", "");
			else
			{
				var parts = line.Trim().Split(new[] { ' ', '(', ')', ',' }, StringSplitOptions.RemoveEmptyEntries);
				if (parts[1] == "float")
					returnType = " returns Number";
				methodName = parts[2];
				for (var index = 3; index < parts.Length; index += 2)
					if (parts[index] == "DepthImage")
					{
						parameters.Add("input Number");
						parameters.Add("Width Number");
						parameters.Add("Height Number");
						parameters.Add("initialDepth Number");
					}
					else if (parts[index] != "float")
						throw new NotSupportedException(parts[index + 1]);
					else
						parameters.Add(parts[index + 1] + " Number");
			}
		}
		if (returnStatement == "")
			throw new MissingReturnStatement();
		var method = new Method(this, 0, new CSharpExpressionParser(),
			new[] { methodName + parameters.ToBrackets() + returnType, "\t" + returnStatement });
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