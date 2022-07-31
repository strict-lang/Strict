using System;
using System.Collections.Generic;
using System.Linq; //TODO: linq should be avoided for better performance

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
		return new NoArgumentMethodCall(constructor, new From(type));//TODO: argument logic, ParseExpression(line, ..));
	}

	public override Expression ParseExpression(Method.Line line, Range rangeToParse)
	{
		var input = line.Text.GetSpanFromRange(rangeToParse);
		if (input.IsEmpty)
			throw new CannotParseEmptyInput(line);
		if (!input.Contains(' ') && !input.Contains(','))
			return Boolean.TryParse(line, rangeToParse) ?? Text.TryParse(line, rangeToParse) ??
				List.TryParseWithSingleElement(line, rangeToParse) ??
				TryParseMemberOrZeroOrOneArgumentMethodCall(line, rangeToParse) ??
				Number.TryParse(line, rangeToParse) ?? (input.IsOperator()
					? throw new InvalidOperatorHere(line, input.ToString())
					: throw new UnknownExpression(line, line.Text[rangeToParse]));
		//check for method call/member call here too
		var postfix = new ShuntingYard(line.Text, rangeToParse);
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

	public Expression? TryParseMemberOrZeroOrOneArgumentMethodCall(Method.Line line, Range range)
	{
		// We can early out here if this looks like a number digit
		if (char.IsNumber(line.Text[range.Start.Value]))
			return null;
		var partToParse = line.Text.GetSpanFromRange(range);
		var argumentsStart = partToParse.IndexOf('(');
		if (argumentsStart > 0)
		{
			var arguments = ParseListArguments(line, argumentsStart + range.Start.Value + 1,
				partToParse.Length - 1);
			if (arguments.Count != 1)
				throw new NotSupportedException("Expected exactly one argument here: " +
					partToParse.ToString());
			partToParse = line.Text.AsSpan(range.Start.Value, argumentsStart);
			if (partToParse.Contains('.'))
			{
				var memberParts = partToParse.Split('.');
				//messed, up call it manually: var member = GetNestedMemberCall(line, memberParts);
				memberParts.MoveNext();
				var firstMemberName = memberParts.Current;
				var first = TryMemberCallOrNoArgumentMethodCall(line, firstMemberName, null);
				if (first == null)
					throw new MemberNotFound(line, line.Method.Type, firstMemberName.ToString());
				memberParts.MoveNext();//should be in a loop obviously!
				var method = first.ReturnType.FindMethod(memberParts.Current.ToString()) ??
					throw new MethodNotFound(line, first + "." + memberParts.Current.ToString(),
						first.ReturnType);
				return new OneArgumentMethodCall(method, first, arguments[0]);
			}
			else
			{
				var method = line.Method.Type.FindMethod(partToParse.ToString());
				if (method != null)
					return new OneArgumentMethodCall(method, null, arguments[0]);
				return null;
			}
		}
		else if (partToParse.Contains('.')) //make sure this is really not a number yo, add a bunch of tests!
		{
			var memberParts = partToParse.Split('.');
			var member = GetNestedMemberCall(line, partToParse.Split('.'));
			return TryMemberCallOrNoArgumentMethodCall(line, memberParts.Current, member);
		}
		else
			return TryMemberCallOrNoArgumentMethodCall(line, partToParse, null);
	}

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
			return new NoArgumentMethodCall(method, null);//TODO, arguments);
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
		var constructorMethodCall = new NoArgumentMethodCall(type.Methods[0], // TODO: Get constructor method using a helper method
			new From(type));//TODO, line.Method.ParseExpression(line, ..)); //TODO: broken anyways: typeNameAndArguments[1]) ?? use same method GetArguments method
		//TODO: this makes no sense!
		if (!hasNestedMethodCall)
			return constructorMethodCall;
		var arguments = typeNameAndArguments.Count > 3
			? GetArguments(line, typeNameAndArguments.Skip(3).ToList())
			: Array.Empty<Expression>();
		var method = type.Methods.FirstOrDefault(m => m.Name == typeNameAndArguments[2][1..]) ?? throw new MethodNotFound(line, typeNameAndArguments[2][1..], type);
		return new NoArgumentMethodCall(method, constructorMethodCall);//TODO, arguments);
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

	private static MemberCall GetNestedMemberCall(Method.Line line, SpanSplitEnumerator partsEnumerator)
	{
		partsEnumerator.MoveNext();
		var firstMemberName = partsEnumerator.Current;
		var first = TryMemberCallOrNoArgumentMethodCall(line, firstMemberName, null);
		if (first == null)
			throw new MemberNotFound(line, line.Method.Type, firstMemberName.ToString());
		//TODO: abort when MoveNext is false, then we are done
		partsEnumerator.MoveNext();
		var secondMemberName = partsEnumerator.Current.ToString();
		//TODO: this whole member thing is a bit strange, because each part can be a member, methodcall, number, whatever!
		var second = //TryMemberCallOrNoArgumentMethodCall(line, secondMemberName, first);
			first.ReturnType.Members.FirstOrDefault(m => m.Name == secondMemberName);
		if (second == null)
			throw new MemberNotFound(line, first.ReturnType, secondMemberName);
		return new MemberCall(first, second);
	}

	private static Expression? TryMemberCallOrNoArgumentMethodCall(Method.Line line, ReadOnlySpan<char> name, MemberCall? memberInstance)
	{
		if (!name.IsWord())
			return null;
		//TODO: this is all scope name checking and should be in its own class
		//TODO: this scope MUST check in the scope of memberInstance, only if it is null we should check here!
		Assignment? foundArgument = null;
		foreach (var variable in line.Method.Variables)
		{
			if (variable is Assignment assignment &&
				name.Equals(assignment.Name.Name, StringComparison.Ordinal))
			{
				foundArgument = assignment;
				break;
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
				if (name.Equals(member.Name, StringComparison.Ordinal))
				{
					foundMember = member;
					break;
				}
			}
		}
		if (foundMember != null)
			return new MemberCall(foundMember);
		//TODO: the member can be anything, any expression, don't assume it is always a member!
		var method = line.Method.Type.FindMethod(name.ToString());
		if (method != null)
			return new NoArgumentMethodCall(method, memberInstance);
		return null;
	}

	public sealed class MemberNotFound : ParsingFailed
	{
		public MemberNotFound(Method.Line line, Type memberType, string memberName) : base(line,
			memberName, memberType) { }
	}

	/// <summary>
	/// Figures out if there are any bracket groups or if there is binary expression action going on.
	/// Could also contain strings, we don't know. Most of the time it will just be a bunch of values.
	/// <see cref="ShuntingYard"/> will only parse till the next comma, has to call this till the end.
	/// </summary>
	public override List<Expression> ParseListArguments(Method.Line line, int start, int end)
	{
		var innerSpan = line.Text.AsSpan(start, end-start);
		if (innerSpan.Contains('(') || innerSpan.Contains('"'))
		{
			// The postfix data comes in upside down, so use another stack to restore order
			var expressions = new Stack<Expression>();
			// Similar to TryParseExpression, but we know there is commas separating things! 
			var postfix = new ShuntingYard(line.Text, new Range(start, end));
			if (postfix.Output.Count == 1)
				expressions.Push(ParseTextWithSpacesOrListWithMultipleOrNestedElements(line,
					postfix.Output.Pop()));
			else if (postfix.Output.Count == 2)
				expressions.Push(Not.Parse(line, postfix));
			else
				do
				{
					var range = postfix.Output.Peek();
					Console.WriteLine("pushing list element "+line.Text[range]);
					var span = line.Text.GetSpanFromRange(range);
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
		return ParseAllElementsFast(line, (start, innerSpan.Length),
			innerSpan.SplitIntoRanges(',', true));
	}

	public class ListTokensAreNotSeparatedByComma : ParsingFailed
	{
		public ListTokensAreNotSeparatedByComma(Method.Line line) : base(line) { }
	}

	private static List<Expression> ParseAllElementsFast(Method.Line line, (int, int) offsetAndLength, RangeEnumerator elements)
	{
		var expressions = new List<Expression>();
		foreach (var element in elements)
			expressions.Add(line.Method.ParseExpression(line, element.GetOuterRange(offsetAndLength)));
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
		var expressions = new List<Expression>();
		for (var lineNumber = 0; lineNumber < method.bodyLines.Count; lineNumber++)
		{
			var expression = ParseMethodLine(method.bodyLines[lineNumber], ref lineNumber);
			if (expression is Assignment assignment)
				method.Variables.Add(assignment);
			expressions.Add(expression);
		}
		//TODO: to clear link to original memory: method.bodyLines = Memory<char>.Empty; //ArraySegment<Method.Line>.Empty;
		return new MethodBody(method, expressions);
	}

	//private readonly PhraseTokenizer tokenizer = new();

	public override Expression ParseMethodLine(Method.Line line, ref int methodLineNumber) =>
		Assignment.TryParse(line) ?? If.TryParse(line, ref methodLineNumber) ??
		//https://deltaengine.fogbugz.com/f/cases/25210
		Return.TryParse(line) ?? ParseExpression(line, ..);
}