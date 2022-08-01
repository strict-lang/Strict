using System;
using System.Collections.Generic;
using System.Linq; //TODO: linq should be avoided for better performance
using System.Linq.Expressions;
using static Strict.Language.Method;

namespace Strict.Language.Expressions;

/// <summary>
/// Parses method bodies by splitting into main lines (lines starting without tabs)
/// and getting the expression recursively via parser combinator logic in each expression.
/// </summary>
public class MethodExpressionParser : ExpressionParser
{
	public override Expression ParseAssignmentExpression(Type type, string initializationLine, int fileLineNumber)
	{
		var constructor = type.Methods[0];
		var line = new Method.Line(constructor, 0, initializationLine, fileLineNumber);
		//TODO: maybe non constructor calls also make sense here?
		return new MethodCall(constructor, new From(type));//TODO: argument logic, ParseExpression(line, ..));
	}

	public override Expression ParseExpression(Method.Line line, Range range)
	{
		var input = line.Text.GetSpanFromRange(range);
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(line);
		if (!input.Contains(' ') && !input.Contains(','))
			return Boolean.TryParse(line, range) ?? Text.TryParse(line, range) ??
				List.TryParseWithSingleElement(line, range) ??
				TryParseMemberOrZeroOrOneArgumentMethodCall(line, range) ??
				Number.TryParse(line, range) ?? (input.IsOperator()
					? throw new InvalidOperatorHere(line, input.ToString())
					: throw new UnknownExpression(line, line.Text[range]));
		//check for method call/member call here too
		var postfix = new ShuntingYard(line.Text, range);
		return postfix.Output.Count switch
		{
			1 => ParseTextWithSpacesOrListWithMultipleOrNestedElements(line, postfix.Output.Pop()),
			//TODO: can also be any method call or anything we excluded above that was still 1 token
			2 => Not.Parse(line, postfix),
			_ => //TODO: should never happen here, Binary will complain if we have a comma there! postfix.Output.Count % 2 != 1 && line.Text[postfix.Output.Skip(1).First().Start.Value] == ',' ?
				Binary.Parse(line, postfix.Output)
		};
		/*from list, same
		 
		var postfixTokens = new ShuntingYard(line.Text, range).Output;
		if (postfixTokens.Count == 1)
			elementsToFill.Add(line.Method.TryParseExpression(line, postfixTokens.Pop()) ??
				throw new MethodExpressionParser.UnknownExpression(line, line.Text[range]));
		else
			foreach (var token in postfixTokens)
			{
				var expressions = new List<Expression>();
				new PhraseTokenizer(line.Text, new Range(start, start + innerSpan.Length)).ProcessEachToken(
					tokenRange =>
					{
						Console.WriteLine("TryParseWithMultipleOrNestedElements: token=" +
							line.Text[tokenRange.Start.Value]);
						if (line.Text[tokenRange.Start.Value] != ',')
							expressions.Add(line.Method.TryParseExpression(line, tokenRange)
								? 7
						throw new MethodExpressionParser.UnknownExpression(line, line.Text[tokenRange]));
					});
			}
		 */
	}

	public sealed class InvalidOperatorHere : ParsingFailed
	{
		public InvalidOperatorHere(Method.Line line, string message) : base(line, message) { }
	}

	public sealed class UnknownExpression : ParsingFailed
	{
		public UnknownExpression(Method.Line line, string error = "") : base(line, error) { }
	}

	public class CannotParseEmptyInput : ParsingFailed
	{
		public CannotParseEmptyInput(Method.Line line) : base(line) { }
	}

	//TODO: error handling (same as constructor calling actually)
	//https://deltaengine.fogbugz.com/f/cases/25211

	/// <summary>
	/// By far the most common usecase, we call something from another instance, use some binary
	/// operator (like is, to, +, etc.) or execute some method. For more arguments more complex
	/// parsing has to be done and we have to invoke ShuntingYard for the argument list.
	/// </summary>
	public Expression? TryParseMemberOrZeroOrOneArgumentMethodCall(Method.Line line, Range range)
	{
		// We can early out here if this looks like a number digit
		if (char.IsNumber(line.Text[range.Start.Value])) //TODO: make sure this is really not a number yo, add a bunch of tests!
			return null;
		var toParse = line.Text.GetSpanFromRange(range);
		var argumentsStart = toParse.IndexOf('(');
		if (argumentsStart > 0)
			return ParseInContext(line.Method.Type, line,
				new Range(range.Start, range.Start.Value + argumentsStart),
				ParseListArguments(line,
					new Range(argumentsStart + range.Start.Value + 1, toParse.Length - 1)));
		return ParseInContext(line.Method.Type, line, range, Array.Empty<Expression>());
	}

	private Expression? ParseInContext(Context context, Method.Line line, Range range, IReadOnlyList<Expression> arguments)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
		Console.WriteLine(nameof(ParseInContext) + " " + context + ", " + partToParse.ToString());
		if (partToParse.Contains('.'))
		{
			var members = new RangeEnumerator(partToParse, '.', range.Start);
			Expression? current = null;
			while (members.MoveNext())
			{
				var expression = TryMemberOrMethodCall(context, current, line, members.Current, arguments);
				if (expression == null)
				{
					// Could also be a Text or List, bool or number are not allowed here in this nested context
					expression = Text.TryParse(line, members.Current) ?? List.TryParseWithSingleElement(line, members.Current);
					if (expression == null)
						throw new MemberOrMethodNotFound(line, current?.ReturnType ?? line.Method.Type,
							line.Text[members.Current]);
				}
				current = expression;
				context = current.ReturnType;
			}
			return current;
		}
		return TryMemberOrMethodCall(context, null, line, range, arguments);
	}

	private static Expression? TryMemberOrMethodCall(Context context, Expression? instance, Method.Line line, Range range,
		IReadOnlyList<Expression> arguments)
	{
		var partToParse = line.Text.GetSpanFromRange(range);
		if (!partToParse.IsWord())
			return null;
		//foreach (var (name, variableValue) in GetAvailableVariables(context))
		//	if (partToParse.Equals(name, StringComparison.Ordinal))
		//		return variableValue;//TODO: should be member yo
		//TODO: test: Find all parent members as well use unit test -> Count(5).Floor is 5
		if (arguments.Count == 0)
		{
			var type = context as Type;
			if (context is Method method)
			{
				foreach (var (name, value) in method.Variables)
					if (partToParse.Equals(name, StringComparison.Ordinal))
						return new VariableCall(name, value);
				foreach (var parameter in method.Parameters)
					if (partToParse.Equals(parameter.Name, StringComparison.Ordinal))
						return new ParameterCall(parameter);
				type = method.ReturnType;
			}
			var memberCall = TryFindMemberCall(type!, instance, partToParse);
			if (memberCall != null)
				return memberCall;
			Console.WriteLine("ParseNested found no member in " + line.Method);
		}
		//TODO: the member can be anything, any expression, don't assume it is always a member!
		//TODO: constructor needed here!
		var method2 = line.Method.Type.FindMethod(partToParse.ToString());
		if (method2 != null)
			return new MethodCall(method2, instance, arguments);
		Console.WriteLine("ParseNested found no local method " + line.Method.Type);
		return null;
	}

	private static Expression? TryFindMemberCall(Type type, Expression? instance, ReadOnlySpan<char> partToParse)
	{
		foreach (var member in type.Members)
			if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				return new MemberCall(instance, member);
		foreach (var implementType in type.Implements)
		{
			var memberCall = TryFindMemberCall(implementType, instance, partToParse);
			if (memberCall != null)
				return memberCall;
		}
		return null;
	}

	/*TODO: remove, would make things easier, but extremely slow! remove after we have a stable code structure!
	/// <summary>
	/// Provides a list of all named members, parameters and local variables with their current values
	/// available in this context scope, all the way recursively down including all implements.
	/// </summary>
	public static Dictionary<NamedType, Expression> GetAvailableVariables(Context context)
	{
		//TODO: optimize! we always only add things, each Context can provide their own implementation as well!
		Dictionary<NamedType, Expression> result = new Dictionary<NamedType, Expression>();
		if (context is Method method)
		{
			foreach (var variable in method.Variables)
			{
				if (variable is Assignment assignment)
					result.Add(assignment.Name.Name, assignment);
			}
			foreach (var parameter in method.Parameters)
			{
				if (variable is Assignment assignment)
					result.Add(parameter.Name, parameter. assignment);
			}
		}
		var foundMember = foundArgument != null
			? new Member(foundArgument.Name.Name, foundArgument.Value)
			: null; // TODO: Find all parent members as well use unit test -> Count(5).Floor is 5
		if (foundMember == null)
		{
			foundMember = null;
			foreach (var member in line.Method.Type.Members)
			{
				if (partToParse.Equals(member.Name, StringComparison.Ordinal))
				{
					foundMember = member;
					break;
				}
			}
		}
		if (foundMember != null)
			return new MemberCall(foundMember);
		Console.WriteLine("ParseNested found no member in " + line.Method);
		//TODO: the member can be anything, any expression, don't assume it is always a member!
		var method2 = line.Method.Type.FindMethod(partToParse.ToString());
		if (method2 != null)
			return new MethodCall(method2, null, arguments);
		Console.WriteLine("ParseNested found no local method " + line.Method.Type);
		return result;
	}
	/*TODO
private Expression GetNestedExpression(Expression? current, Method.Line line, Range range, IReadOnlyList<Expression> arguments)
{
	Console.WriteLine("GetNestedExpression: current="+current+", text="+line.Text[range]);
	return TryMemberOrMethodCall( //TODO?current, should all be in scope class
			line, line.Text.GetSpanFromRange(range), arguments) ??
		throw new MemberOrMethodNotFound(line, line.Method.Type, line.Text[range]);
	var first = TryMemberCallOrNoArgumentMethodCall(line, firstMemberName, null);
	if (first == null)
		throw new MemberNotFound(line, line.Method.Type, firstMemberName.ToString());
	
	var method = first.ReturnType.FindMethod(memberParts.Current.ToString()) ??
		throw new MethodNotFound(line, first + "." + memberParts.Current.ToString(),
			first.ReturnType);
	return new OneArgumentMethodCall(method, first, arguments[0]);


	while (members.MoveNext())
	{
		//TODO: abort when MoveNext is false, then we are done
		partsEnumerator.MoveNext();
		var secondMemberName = partsEnumerator.Current.ToString();
		//TODO: this whole member thing is a bit strange, because each part can be a member, methodcall, number, whatever!
		var second = //TryMemberCallOrNoArgumentMethodCall(line, secondMemberName, first);
			first.ReturnType.Members.FirstOrDefault(m => m.Name == secondMemberName);
		if (second == null)
			throw new MemberNotFound(line, first.ReturnType, secondMemberName);
		var member = new MemberCall(first, second);
		return TryMemberCallOrNoArgumentMethodCall(line, members.Current, member);
	//return null!;
}
	*/

	private static Expression? TryParseMethod(Method.Line line, Range range, string[] parts)
	{
		var methodName = parts[0];
		var argumentStartIndex = range.Start.Value + parts[0].Length + 1;
		Expression[] arguments;
		if (parts.Length > 1)
			arguments = GetArguments(line, parts[1], methodName, argumentStartIndex);
		else
			arguments = Array.Empty<Expression>();
		if (!methodName.Contains('.'))
			// get the method
		{
			var method = FindMethod(null, line, methodName, arguments);
			if (method == null)
				//TODO: If not found check types for constructor call may be inline it here?
			{
				return
					TryParseFrom(line,
						range); //range.Start.Value..(methodName.Length + range.Start.Value));
			}
			return new MethodCall(method, null, arguments);
		}
		/*TODO: should be way more generic, it is not just a nested member call, it can be anything!
		var memberParts = methodName.Split('.', 2);
		var firstMember = MemberCall.TryParse(line, range.Start..memberParts[0].Length);
		if (firstMember == null)
			throw new MemberCall.MemberNotFound(line, line.Method.Type, memberParts[0]);
		var memberMethod = FindMethod(firstMember, line, memberParts[1], arguments);
		return memberMethod != null
			? new NoArgumentMethodCall(memberMethod, firstMember)//TODO, arguments)
			: throw new MethodNotFound(line, memberParts[1], firstMember.ReturnType); // TODO: still check for members
		*/
		throw new NotSupportedException("TODO");
	}

	//TODO: should correctly find method and call the right number of argument MethodCall
	private static Expression[] GetArguments(Method.Line line, string argumentsText,
		string methodName, int argumentStartIndex)
	{
		// someClass.ComplicatedMethod((1, 2, 3) + (4, 5), 7)
		// list of 2 arguments:
		// [0] = (1, 2, 3) + (4, 5)
		// [1] = 7
		// don't use this, broken, we already have working list parsing
		var parts = argumentsText.Split(", ");
		var arguments = new Expression[parts.Length];
		for (var index = 0; index < parts.Length; index++)
			try
			{
				arguments[index] = line.Method.ParseExpression(line,
					argumentStartIndex..(argumentStartIndex + parts[index].Length));
			}
			catch (MethodExpressionParser.UnknownExpression)
			{
				throw new InvalidExpressionForArgument(line, parts[index] + " is invalid for " + methodName + " argument " + index);
			}
		return arguments;
	}

	public sealed class InvalidExpressionForArgument : ParsingFailed
	{
		public InvalidExpressionForArgument(Method.Line line, string message) : base(line, message) { }
	}

	// ReSharper disable once TooManyArguments
	private static Method? FindMethod(Expression? instance, Method.Line line, string methodName,
		Expression[] arguments)
	{
		if (!methodName.IsWord())
			return null;
		var context = instance?.ReturnType ?? line.Method.Type;
		var method = context.Methods.FirstOrDefault(m => m.Name == methodName);
		return method; /* TODO: add once constructor is fixed == null
			? throw new MethodNotFound(line, methodName, context)
			: method.Parameters.Count != arguments.Length
				? throw new ArgumentsDoNotMatchMethodParameters(line, arguments, method)
				: method;*/
	}

	public sealed class MethodNotFound : ParsingFailed
	{
		public MethodNotFound(Method.Line line, string methodName, Type referencingType) : base(line, methodName, referencingType) { }
	}

	public sealed class ArgumentsDoNotMatchMethodParameters : ParsingFailed
	{
		public ArgumentsDoNotMatchMethodParameters(Method.Line line, Expression[] arguments,
			Method method) : base(line, (arguments.Length == 0
				? "No arguments does "
				: "Arguments: " + arguments.ToBrackets() + " do ") + "not match \"" + method.Type + "." +
			method.Name + "\" method parameters: " + method.Parameters.ToBrackets()) { }
	}

	public static Expression? TryParseFrom(Method.Line line, Range range)
	{
		var partToParse = line.Text[range]; //TODO: use span here!
		return partToParse.EndsWith(')') && partToParse.Contains('(')
			? TryParseFrom(line,
				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries),
				partToParse.Contains(")."))
			: null;
	}

	private static Expression? TryParseFrom(Method.Line line,
		IReadOnlyList<string> typeNameAndArguments, bool hasNestedMethodCall)
	{
		var type = line.Method.FindType(typeNameAndArguments[0]);
		if (type == null)
			return null;
		var constructorMethodCall = new MethodCall(type.Methods[0], // TODO: Get constructor method using a helper method
			new From(type), Array.Empty<Expression>());//TODO, line.Method.ParseExpression(line, ..)); //TODO: broken anyways: typeNameAndArguments[1]) ?? use same method GetArguments method
		//TODO: this makes no sense!
		if (!hasNestedMethodCall)
			return constructorMethodCall;
		var arguments = typeNameAndArguments.Count > 3
			? GetArguments(line, typeNameAndArguments.Skip(3).ToList())
			: Array.Empty<Expression>();
		var method = type.Methods.FirstOrDefault(m => m.Name == typeNameAndArguments[2][1..]) ?? throw new MethodNotFound(line, typeNameAndArguments[2][1..], type);
		return new MethodCall(method, constructorMethodCall, arguments);
	}

	private static Expression[] GetArguments(Method.Line line, IReadOnlyList<string> parts)
	{
		var arguments = new Expression[parts.Count];
		for (var index = 0; index < parts.Count; index++)
			//TODO: this is the same as above
			try
			{
				arguments[index] = line.Method.ParseExpression(line, ..); //TODO: parts[index]) ??
			}
			catch (MethodExpressionParser.UnknownExpression)
			{ //TODO: this is duplicated code!
				throw new InvalidExpressionForArgument(line,
					parts[index] + " is invalid for " + parts[index] + " argument " + index);
			}
		return arguments;
	}
//using System;
//using System.Collections.Generic;
//using System.Linq;

//namespace Strict.Language.Expressions;

//public class Constructor //TODO: merge with normal method call
//{
//	public static Expression? TryParse(Method.Line line, Range range)
//	{
//		var partToParse = line.Text[range];//TODO: use span here!
//		return partToParse.EndsWith(')') && partToParse.Contains('(')
//			? TryParseConstructor(line,
//				partToParse.Split(new[] { '(', ')' }, StringSplitOptions.RemoveEmptyEntries),
//				partToParse.Contains(")."))
//			: null;
//	}

//	private static Expression? TryParseConstructor(Method.Line line,
//		IReadOnlyList<string> typeNameAndArguments, bool hasNestedMethodCall)
//	{
//		var type = line.Method.FindType(typeNameAndArguments[0]);
//		if (type == null)
//			return null;
//		var constructorMethodCall = new MethodCall(new Value(type, type), type.Methods[0],
//			line.Method.TryParseExpression(line, ..) ??//TODO: broken anyways: typeNameAndArguments[1]) ??
//			throw new MethodExpressionParser.UnknownExpression(line));
//		if (!hasNestedMethodCall)
//			return constructorMethodCall;
//		var arguments = typeNameAndArguments.Count > 3
//			? GetArguments(line, typeNameAndArguments.Skip(3).ToList())
//			: Array.Empty<Expression>();
//		var method = type.Methods.FirstOrDefault(m => m.Name == typeNameAndArguments[2][1..]) ?? throw new MethodCall.MethodNotFound(line, typeNameAndArguments[2][1..], type);
//		return new MethodCall(constructorMethodCall, method, arguments);
//	}

//	private static Expression[] GetArguments(Method.Line line, IReadOnlyList<string> parts)
//	{
//		var arguments = new Expression[parts.Count];
//		for (var index = 0; index < parts.Count; index++)
//			arguments[index] = line.Method.TryParseExpression(line, ..) ??//TODO: parts[index]) ??
//				throw new MethodCall.InvalidExpressionForArgument(line,
//					parts[index] + " for " + parts[index] + " argument " + index);
//		return arguments;
//	}
//}

public sealed class MemberOrMethodNotFound : ParsingFailed
	{
		public MemberOrMethodNotFound(Method.Line line, Type memberType, string memberName) : base(line,
			memberName, memberType) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard"/> will only parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Method.Line line, Range range)
	{
		var innerSpan = line.Text.GetSpanFromRange(range);
		if (innerSpan.Contains('(') || innerSpan.Contains('"'))
		{
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things! 
			var postfix = new ShuntingYard(line.Text, range);
			if (postfix.Output.Count == 1)
				expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(line,
					postfix.Output.Pop()));
			else if (postfix.Output.Count == 2)
				expressions.Push(Not.Parse(line, postfix));
			else
				do
				{
					Console.WriteLine("pushing list element "+line.Text[postfix.Output.Peek()]);
					var span = line.Text.GetSpanFromRange(postfix.Output.Peek());
					// Is this a binary expression we have to put into the list (already tokenized and postfixed)
					if (span.Length == 1 && span[0].IsSingleCharacterOperator() ||
						span.IsMultiCharacterOperator())
						expressions.Push(Binary.Parse(line, postfix.Output));
					else
						expressions.Push(line.Method.ParseExpression(line, postfix.Output.Pop()));
					if (postfix.Output.Count > 0 && line.Text[postfix.Output.Pop().Start.Value] != ',')
						throw new ListTokensAreNotSeparatedByComma(line);
				} while (postfix.Output.Count > 0);
				//postfix.Output.Count % 2 != 1 &&
				//line.Text[postfix.Output.Skip(1).First().Start.Value] == ',' ?;}
				//do
				//{
				//	start = end - remainingToParse;
				//} while (start < end);
			return expressions.ToList();
		}
		return ParseAllElementsFast(line, new RangeEnumerator(innerSpan, ',', range.Start));
	}

	public class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Method.Line line) : base(line) { }
	}

	private static List<Expression> ParseAllElementsFast(Method.Line line, RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			expressions.Add(line.Method.ParseExpression(line, element));
		return expressions;
	}

	//TODO: Probably not needed
	public static bool HasIncompatibleDimensions(Expression left, Expression right) =>
		left is List leftList && right is List rightList &&
		leftList.Values.Count != rightList.Values.Count;

	//TODO: as discussed in meeting, we use generics and always check if the right side is castable into the left side (via from), e.g. make a test where we add a Count to a list of Texts -> output list of texts (always from left side), we never change the left side type
	public static bool HasMismatchingTypes(Expression left, Expression right) =>
		left is List leftList && !leftList.IsFirstType<Text>() && right switch
		{
			List rightList when rightList.IsFirstType<Text>() => true,
			Binary { Instance: List rightBinaryLeftList } when rightBinaryLeftList.IsFirstType<Text>() =>
				true,
			Binary { Instance: Text } => true,
			_ => !leftList.IsFirstType<Text>() && right is Text
		};

	public sealed class ListsHaveDifferentDimensions : ParsingFailed
	{
		public ListsHaveDifferentDimensions(Method.Line line, string error) : base(line, error) { }
	}

	private static Expression
		ParseTextWithSpacesOrListWithMultipleOrNestedElements(Method.Line line, Range range) =>
		Text.TryParse(line, range) ?? List.TryParseWithMultipleOrNestedElements(line, range) ??
		throw new InvalidSingleTokenExpression(line, line.Text[range]);

	private sealed class InvalidSingleTokenExpression : ParsingFailed
	{
		public InvalidSingleTokenExpression(Method.Line line, string message) : base(line, message) { }
	}

	/// <summary>
	/// Called lazily by Method.Body and only if needed for execution (context should be over there
	/// as parsing is done in parallel, we should not keep any state here).
	/// </summary>
	public override Expression ParseMethodBody(Method method)
	{
		if (method.bodyLines.Count == 0)
			return new MethodBody(method, Array.Empty<Expression>());
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
			expressions.Add(ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber));
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//https://deltaengine.fogbugz.com/f/cases/25210
		Return.TryParse(line) ?? ParseExpression(line, ..);
}