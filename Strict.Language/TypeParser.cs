using System.Globalization;

namespace Strict.Language;

public sealed class TypeParser(Type type, string[] lines)
{
	private string[] lines = lines;

	public void ParseMembersAndMethods(ExpressionParser parser)
	{
		for (LineNumber = 0; LineNumber < lines.Length; LineNumber++)
			TryParse(parser, LineNumber);
	}

	private void TryInitializeMemberInitialValues(ExpressionParser parser)
	{
		try
		{
			foreach (var pair in rememberToInitializeMemberInitialValues!)
				pair.Key.InitialValue = GetMemberExpression(parser, pair.Key.Name, pair.Value, pair.Key.LineNumber);
		}
		catch (ParsingFailed)
		{
			type.Dispose();
			throw;
		}
		catch (Exception ex)
		{
			type.Dispose();
			throw new ParsingFailed(type, 0, string.IsNullOrEmpty(ex.Message)
				? ex.GetType().Name
				: ex.Message, ex);
		}
	}

	/// <summary>
	/// Should be a property, but that is way slower in debug mode when this is most useful!
	/// </summary>
	internal int LineNumber = -1;

	private void TryParse(ExpressionParser parser, int rememberStartMethodLineNumber)
	{
		try
		{
			ParseLineForMembersAndMethods(parser);
		}
		catch (Context.TypeNotFound ex)
		{
			type.Dispose();
			throw new ParsingFailed(type, rememberStartMethodLineNumber, ex.Message, ex);
		}
		catch (ParsingFailed)
		{
			type.Dispose();
			throw;
		}
		catch (Exception ex)
		{
			type.Dispose();
			throw new ParsingFailed(type, rememberStartMethodLineNumber,
				string.IsNullOrEmpty(ex.Message)
					? ex.GetType().Name
					: ex.Message, ex);
		}
	}

	private void ParseLineForMembersAndMethods(ExpressionParser parser)
	{
		var line = ValidateCurrentLineIsNonEmptyAndTrimmed();
		if (line.StartsWith(Type.HasWithSpaceAtEnd, StringComparison.Ordinal))
			type.Members.Add(GetNewMember(parser));
		else if (line.StartsWith(Type.MutableWithSpaceAtEnd, StringComparison.Ordinal) &&
			!(LineNumber + 1 < lines.Length && lines[LineNumber + 1].StartsWith('\t')))
			type.Members.Add(GetNewMember(parser, Keyword.Mutable));
		else if (line.StartsWith(Type.ConstantWithSpaceAtEnd, StringComparison.Ordinal))
			type.Members.Add(GetNewMember(parser, Keyword.Constant));
		else
		{
			var methodFirstLineNumber = LineNumber;
			var methodLines = GetAllMethodLines(methodFirstLineNumber);
			DetectTrivialEndlessRecursionInFrom(methodLines);
			DetectSelfRecursionWithSameArguments(methodLines);
			DetectHugeConstantRange(methodLines);
			var method = new Method(type, methodFirstLineNumber, parser, methodLines);
			DetectRedundantReturn(methodLines, method);
			var existingMethod = type.Methods.Find(m =>
				m.Name == method.Name && m.ReturnType == method.ReturnType && m.Parameters.
					Select(p => p.Type).SequenceEqual(method.Parameters.Select(p => p.Type)));
			if (existingMethod != null)
				throw new MethodWithSameNameAndParameterCountAlreadyExists(type, //ncrunch: no coverage
					methodFirstLineNumber, method, existingMethod);
			type.Methods.Add(method);
		}
	}

	public sealed class MethodWithSameNameAndParameterCountAlreadyExists(Type type,
		int lineNumber, Method method, Method existingMethod)
		: ParsingFailed(type, lineNumber, method.ToString(), existingMethod.ToString()); //ncrunch: no coverage

	/// <summary>
	/// If a from(...) method contains a same-type constructor call like TypeName(constant) and the
	/// call's argument does not reference any parameter, it will just recursively call itself
	/// forever (e.g., Character.from used Character(0), which would forever call itself).
	/// </summary>
	private void DetectTrivialEndlessRecursionInFrom(IReadOnlyList<string> methodLines)
	{
		if (methodLines.Count == 0)
			return; //ncrunch: no coverage
		var signature = methodLines[0];
		var openParen = signature.IndexOf('(');
		if (openParen <= 0)
			return;
		var methodName = signature[..openParen];
		if (!methodName.Equals(Method.From, StringComparison.Ordinal))
			return;
		var paramNames = CollectParameterNamesFromSignature(signature, openParen);
		// Inspect body lines (all following lines), skip inline tests (containing "is")
		for (var i = 1; i < methodLines.Count; i++)
		{
			var line = methodLines[i];
			if (IsNonTestMethodLine(line))
				continue;
			var typeCtorPrefix = type.Name + "(";
			var idx = line.IndexOf(typeCtorPrefix, StringComparison.Ordinal);
			if (idx < 0)
				continue;
			// Extract argument inside TypeName(...)
			var startArgs = idx + typeCtorPrefix.Length;
			var endArgs = line.IndexOf(')', startArgs);
			if (endArgs <= startArgs)
				continue; //ncrunch: no coverage
			var argText = line[startArgs..endArgs];
			// If argText does not contain any parameter name, it's a constant/self-call -> flag
			var usesAnyParam = paramNames.Any(p => argText.Contains(p, StringComparison.Ordinal));
			if (!usesAnyParam)
				throw new TrivialEndlessSelfConstructionDetected(type, LineNumber, line.Trim());
		}
	}

	private static bool IsNonTestMethodLine(string line) =>
		!line.StartsWith('\t') || line.Contains(" is ", StringComparison.Ordinal);

	private static HashSet<string> CollectParameterNamesFromSignature(string signature, int openParen)
	{
		var closeParen = signature.IndexOf(')', openParen + 1);
		var paramNames = new HashSet<string>(StringComparer.Ordinal);
		if (closeParen > openParen + 1)
		{
			var inside = signature[(openParen + 1)..closeParen];
			foreach (var param in inside.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
			{
				// the param format is "name Type" or just "name"
				var parts = param.Split(' ', StringSplitOptions.RemoveEmptyEntries);
				if (parts.Length > 0)
					paramNames.Add(parts[0]);
			}
		}
		return paramNames;
	}

	/// <summary>
	/// General rule for any method: calling ourselves with the same parameter list (e.g., Foo(a, b))
	/// is a guaranteed endless recursion.
	/// </summary>
	private void DetectSelfRecursionWithSameArguments(IReadOnlyList<string> methodLines)
	{
		if (methodLines.Count == 0 || type.Name == Type.System)
			return;
		var signature = methodLines[0];
		var openParen = signature.IndexOf('(');
		var closeParen = signature.IndexOf(')', openParen + 1);
		if (openParen <= 0 || closeParen <= openParen)
			return;
		var methodName = signature[..openParen].Trim();
		var paramNames = GetParameterNames(signature, openParen, closeParen);
		if (paramNames.Count != 0)
			for (var i = 1; i < methodLines.Count; i++)
				if (!IsNonTestMethodLine(methodLines[i]))
					SearchForMethodCalls(methodName, signature, methodLines[i], paramNames);
	}

	private static List<string> GetParameterNames(string signature, int openParen, int closeParen)
	{
		var paramNames = new List<string>();
		var inside = signature[(openParen + 1)..closeParen];
		foreach (var param in inside.Split(',',
			StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
		{
			var parts = param.Split(' ', StringSplitOptions.RemoveEmptyEntries);
			if (parts.Length > 0)
				paramNames.Add(parts[0]);
		}
		return paramNames;
	}

	private void SearchForMethodCalls(string methodName, string signature, string line, IReadOnlyList<string> paramNames)
	{
		var searchStart = 0;
		var directPattern = methodName + "(";
		while (true)
		{
			var directIdx = line.IndexOf(directPattern, searchStart, StringComparison.Ordinal);
			if (directIdx < 0)
				break;
			// Ensure it's not part of an identifier or a member call (preceded by '.' or word char)
			var prevCharIdx = directIdx - 1;
			if (prevCharIdx >= 0)
			{
				var prev = line[prevCharIdx];
				if (prev == '.' || prev.IsLetter())
				{
					searchStart = directIdx + directPattern.Length;
					continue;
				}
			}
			var argsStartDirect = directIdx + directPattern.Length - 1; // at '('
			var argsEndDirect = line.IndexOf(')', argsStartDirect + 1);
			if (argsEndDirect > argsStartDirect)
			{
				var argText = line[(argsStartDirect + 1)..argsEndDirect];
				if (AreParametersEqual(argText, paramNames))
					throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber, signature, argText, line.Trim());
			}
			searchStart = directIdx + directPattern.Length;
		}
		SearchForMemberMethodCalls(methodName, signature, line, paramNames);
	}

	private static bool AreParametersEqual(string argText, IReadOnlyList<string> paramNames)
	{
		var argNames = argText.Split(',',
			StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
		if (argNames.Length != paramNames.Count)
			return false; //ncrunch: no coverage
		for (var i = 0; i < argNames.Length; i++)
			if (!argNames[i].Equals(paramNames[i], StringComparison.Ordinal))
				return false;
		return true;
	}

	private void SearchForMemberMethodCalls(string methodName, string signature, string line, IReadOnlyList<string> paramNames)
	{
		var searchStart = 0;
		var dotPattern = "." + methodName + "(";
		while (true)
		{
			var dotIdx = line.IndexOf(dotPattern, searchStart, StringComparison.Ordinal);
			if (dotIdx < 0)
				break;
			var receiverEnd = dotIdx - 1;
			var receiverStart = receiverEnd;
			while (receiverStart >= 0 && line[receiverStart].IsLetter())
				receiverStart--;
			receiverStart++;
			var receiver = receiverStart <= receiverEnd
				? line.Substring(receiverStart, receiverEnd - receiverStart + 1)
				: string.Empty;
			// Only treat as recursion if calling this.Method(...) or TypeName.Method(...)
			if (receiver.Equals("this", StringComparison.Ordinal) ||
				receiver.Equals(type.Name, StringComparison.Ordinal))
				CheckRecursionCallingThisMethod(signature, line, paramNames, dotIdx, dotPattern);
			searchStart = dotIdx + dotPattern.Length;
		}
	}

	private void CheckRecursionCallingThisMethod(string signature, string line, IReadOnlyList<string> paramNames, int dotIdx, string dotPattern)
	{
		var argsStart = dotIdx + dotPattern.Length - 1;
		var argsEnd = line.IndexOf(')', argsStart + 1);
		if (argsEnd > argsStart)
		{
			var argText = line[(argsStart + 1)..argsEnd];
			if (AreParametersEqual(argText, paramNames))
				throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber, signature, argText, line.Trim());
		} //ncrunch: no coverage
	} //ncrunch: no coverage

	/// <summary>
	/// Prevent obviously gigantic constant ranges like 1 billion, no need for that in Strict.
	/// </summary>
	private void DetectHugeConstantRange(IReadOnlyList<string> methodLines)
	{
		const long MaximumRangeAllowed = 1_000_000_000L;
		for (var i = 1; i < methodLines.Count; i++)
		{
			var line = methodLines[i];
			var idx = line.IndexOf("Range(", StringComparison.Ordinal);
			if (IsNonTestMethodLine(line) || idx < 0)
				continue;
			var startArgs = idx + "Range(".Length;
			var endArgs = line.IndexOf(')', startArgs);
			var args = line[startArgs..endArgs].Split(',',
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
			if (args.Length == 2 &&
				long.TryParse(args[0], NumberStyles.Integer, CultureInfo.InvariantCulture,
					out var start) && long.TryParse(args[1], NumberStyles.Integer,
					CultureInfo.InvariantCulture, out var end))
			{
				var span = Math.Abs(end - start);
				if (span > MaximumRangeAllowed)
					throw new HugeConstantRangeNotAllowed(type, LineNumber, line.Trim(), span,
						MaximumRangeAllowed);
			} //ncrunch: no coverage
		}
	}

	public sealed class TrivialEndlessSelfConstructionDetected(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber,
			"Endless recursion via self-constructor call in from: " + line);

	public sealed class SelfRecursiveCallWithSameArgumentsDetected(Type type, int lineNumber,
		string signature, string argumentNames, string line) : ParsingFailed(type, lineNumber,
		$"Self-recursive call with same arguments detected in {
			signature
		} with arguments=({
			GetArgumentTypes(signature)
		}) called with ({
			argumentNames
		}): {
			line
		}")
	{
		private static string GetArgumentTypes(string signature) =>
			string.Join(", ",
				signature[(signature.IndexOf('(') + 1)..signature.IndexOf(')')].
					Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).
					Select(parameter =>
						parameter.Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).
							FirstOrDefault() ?? Type.Any));
	}

	public sealed class HugeConstantRangeNotAllowed(Type type, int lineNumber, string line,
		long span, long limit) : ParsingFailed(type, lineNumber,
		$"Range size {span} exceeds limit {limit}: " + line);

	private void DetectRedundantReturn(IReadOnlyList<string> checkLines, Method method)
	{
		if (checkLines[^1].StartsWith("\treturn ", StringComparison.Ordinal))
			throw new Body.ReturnAsLastExpressionIsNotNeeded(new Body(method));
		if (checkLines.Count < 3)
			return;

		//TODO: way to complicated and slow just to check for /t at the beginning of a line
		static int GetIndent(string line)
		{
			return line.TakeWhile(c => c == '\t').Count();
		}

		if (GetIndent(checkLines[^1]) != GetIndent(checkLines[^2]))
			return;
		var prevAssignmentIndex = checkLines[^2].IndexOf(" = ", StringComparison.Ordinal);
		if (prevAssignmentIndex <= 0)
			return;
		//TODO: isn't this all a bit complicated and using too many sub strings (creation) and linq methods that can be avoided?
		var left = checkLines[^2][1..prevAssignmentIndex];
		var right = checkLines[^2][(prevAssignmentIndex + 3)..];
		var variableName = left.Split(' ', StringSplitOptions.RemoveEmptyEntries).LastOrDefault();
		if (!string.IsNullOrEmpty(variableName) &&
			//TODO: shouldn't be needed, this is a slow check
			(string.Equals(checkLines[^1].TrimStart(), variableName, StringComparison.Ordinal) ||
				string.Equals(checkLines[^1].TrimStart(), right, StringComparison.Ordinal)))
			throw new RedundantReturnPreviousLineContainsValueAlready(type, LineNumber, checkLines[^2],
				variableName);
	}

	public sealed class RedundantReturnPreviousLineContainsValueAlready(Type type, int lineNumber,
		string prevLine, string variableName) : ParsingFailed(type, lineNumber, prevLine, variableName);

	private string ValidateCurrentLineIsNonEmptyAndTrimmed()
	{
		var line = lines[LineNumber];
		if (line.Length == 0)
			throw new EmptyLineIsNotAllowed(type, LineNumber);
		return char.IsWhiteSpace(line[0])
			? throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line)
			: char.IsWhiteSpace(line[^1])
				? throw new ExtraWhitespacesFoundAtEndOfLine(type, LineNumber, line)
				: line;
	}

	public sealed class EmptyLineIsNotAllowed(Type type, int lineNumber)
		: ParsingFailed(type, lineNumber);

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine(Type type, int lineNumber,
		string message, string method = "") : ParsingFailed(type, lineNumber,
		message + " (strict always requires tab for indentation)", method);

	public sealed class ExtraWhitespacesFoundAtEndOfLine(Type type, int lineNumber,
		string message, string method = "") : ParsingFailed(type, lineNumber, message, method);

	private Member GetNewMember(ExpressionParser parser, string usedKeyword = Keyword.Has)
	{
		var member = ParseMember(parser, lines[LineNumber].AsSpan(usedKeyword.Length + 1),
			usedKeyword);
		return type.Members.Any(m => m.Name == member.Name)
			? throw new DuplicateMembersAreNotAllowed(type, LineNumber, member.Name)
			: member;
	}

	private Member ParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		string usedKeyword)
	{
		if (type.Methods.Count > 0)
			throw new MembersMustComeBeforeMethods(type, LineNumber, remainingLine.ToString());
		try
		{
			return TryParseMember(parser, remainingLine, usedKeyword);
		}
		catch (ParsingFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
			throw new ParsingFailed(type, LineNumber, ex.Message.Split('\n').Take(2).ToLines(), ex);
		}
	}

	private Member TryParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		string usedKeyword)
	{
		var nameAndExpression = remainingLine.Split();
		nameAndExpression.MoveNext();
		var nameAndType = nameAndExpression.Current.ToString();
		if (nameAndExpression.MoveNext())
		{
			var wordAfterName = nameAndExpression.Current.ToString();
			if (nameAndExpression.Current[0] == EqualCharacter)
			{
				var constantValue = remainingLine[(nameAndType.Length + 3)..];
				var withIndex = constantValue.IndexOf(" " + Keyword.With + " ", StringComparison.Ordinal);
				var valueOnly = withIndex >= 0
					? constantValue[..withIndex]
					: constantValue;
				if (usedKeyword == Keyword.Constant &&
					int.TryParse(valueOnly, out var forcedEnumNumber))
					type.AutogeneratedEnumValue = forcedEnumNumber;
				var member = new Member(type, nameAndType,
					GetInitialValueType(parser, nameAndType, valueOnly), LineNumber, usedKeyword)
				{
					InitialValueText = valueOnly.ToString()
				};
				rememberToInitializeMemberInitialValues ??= new Dictionary<Member, string>();
				rememberToInitializeMemberInitialValues.Add(member, valueOnly.ToString());
				if (withIndex >= 0)
				{
					var constraintsSpan = constantValue[(withIndex + Keyword.With.Length + 2)..];
					pendingConstraints ??= new List<(Member, string[])>();
					pendingConstraints.Add((member,
						constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries)));
				}
				return member;
			}
			if (wordAfterName != Keyword.With && wordAfterName != Keyword.Where)
			{
				var explicitType = GetMemberType(nameAndExpression);
				if (string.Equals(nameAndType, explicitType, StringComparison.Ordinal) ||
					string.Equals(nameAndType.MakeFirstLetterUppercase(), explicitType,
						StringComparison.Ordinal))
					throw new RedundantExplicitMemberTypeName(type, LineNumber, nameAndType,
						explicitType);
				nameAndType += " " + explicitType;
			}
			if (HasConstraints(wordAfterName, ref nameAndExpression))
				return !nameAndExpression.MoveNext()
					? throw new MemberMissingConstraintExpression(type, LineNumber, nameAndType)
					: IsMemberTypeAny(nameAndType, nameAndExpression)
						? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
						: GetMemberWithConstraints(parser, remainingLine, usedKeyword, nameAndType);
			if (nameAndExpression.Current[0] == EqualCharacter)
				return ParseTypedMemberWithInitialValue(parser, remainingLine, usedKeyword, nameAndType);
		}
		return IsMemberTypeAny(nameAndType, nameAndExpression)
			? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
			: usedKeyword == Keyword.Constant
				? new Member(type, nameAndType, type.GetType(Type.Number), LineNumber, usedKeyword)
				{
					InitialValue = GetMemberExpression(parser, nameAndType,
						(type.AutogeneratedEnumValue++).ToString(), LineNumber)
				}
				: new Member(type, nameAndType, null, LineNumber, usedKeyword);
	}

	private Type GetInitialValueType(ExpressionParser parser, string nameAndType,
		ReadOnlySpan<char> constantValue)
	{
		if (constantValue.EndsWith(')'))
		{
			var openBracketIndex = constantValue.IndexOf('(');
			if (openBracketIndex > 0)
			{
				var initialValueTypeName = constantValue[..openBracketIndex];
				return initialValueTypeName == type.Name
					? type
					: type.GetType(initialValueTypeName.ToString());
			}
		}
		var memberNameWithFirstLetterCaps = nameAndType.MakeFirstLetterUppercase();
		var memberNameType = type.FindType(memberNameWithFirstLetterCaps);
		if (memberNameType != null && !constantValue.StartsWith(memberNameWithFirstLetterCaps))
			return memberNameType;
		if (constantValue.StartsWith('\"'))
			return type.GetType(Type.Text);
		if (constantValue is "true" || constantValue is "false")
			return type.GetType(Type.Boolean); //ncrunch: no coverage
		return constantValue.TryParseNumber(out _)
			? type.GetType(Type.Number)
			: GetMemberExpression(parser, nameAndType, constantValue, LineNumber).ReturnType;
	}

	private Dictionary<Member, string>? rememberToInitializeMemberInitialValues;

	/// <summary>
	/// Handles "has Name Type = value" syntax where both explicit type and initial value are given.
	/// </summary>
	private Member ParseTypedMemberWithInitialValue(ExpressionParser parser,
		ReadOnlySpan<char> remainingLine, string usedKeyword, string nameAndType)
	{
		var equalIndex = remainingLine.IndexOf(" = ");
		var constantValue = remainingLine[(equalIndex + 3)..];
		var withIndex = constantValue.IndexOf(" " + Keyword.With + " ", StringComparison.Ordinal);
		var valueOnly = withIndex >= 0
			? constantValue[..withIndex]
			: constantValue;
		var member = new Member(type, nameAndType, null, LineNumber, usedKeyword)
   {
			InitialValueText = GetTypedMemberInitializerText(parser, nameAndType, valueOnly)
		};
		rememberToInitializeMemberInitialValues ??= new Dictionary<Member, string>();
    rememberToInitializeMemberInitialValues.Add(member, member.InitialValueText);
		if (withIndex >= 0)
		{
			var constraintsSpan = constantValue[(withIndex + Keyword.With.Length + 2)..];
			pendingConstraints ??= new List<(Member, string[])>();
			pendingConstraints.Add((member,
				constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries)));
		}
		return member;
	}

	private string GetTypedMemberInitializerText(ExpressionParser parser, string nameAndType,
		ReadOnlySpan<char> valueOnly)
	{
		var declaredTypeName = nameAndType[(nameAndType.IndexOf(' ') + 1)..];
		if (StartsWithConstructorCall(valueOnly, declaredTypeName))
			throw new NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithSameType(nameAndType);
		var parsedValue = parser.ParseExpression(
			new Body(new Method(type, LineNumber, parser, [nameof(GetMemberExpression)])), valueOnly);
   return parsedValue.ReturnType.Name == declaredTypeName ||
			!HasMatchingFromConstructor(type.GetType(declaredTypeName), parsedValue.ReturnType)
				? valueOnly.ToString()
				: declaredTypeName + "(" + valueOnly.ToString() + ")";
	}

	private static bool HasMatchingFromConstructor(Type declaredType, Type valueType) =>
		declaredType.AvailableMethods.TryGetValue(Method.From, out var fromMethods) &&
		fromMethods.Any(fromMethod => fromMethod.Parameters.Count == 1 &&
			valueType.IsSameOrCanBeUsedAs(fromMethod.Parameters[0].Type));

	private static bool StartsWithConstructorCall(ReadOnlySpan<char> valueOnly, string declaredTypeName) =>
		valueOnly.StartsWith(declaredTypeName, StringComparison.Ordinal) &&
		valueOnly.Length > declaredTypeName.Length && valueOnly[declaredTypeName.Length] == '(';

	private Member GetMemberWithConstraints(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		string usedKeyword, string nameAndType)
	{
		var member = new Member(type, nameAndType,
			ExtractConstraintsSpanAndValueType(parser, remainingLine, nameAndType,
				out var constraintsSpan, out var initialValueSpan), LineNumber, usedKeyword);
		if (initialValueSpan != "")
		{
			member.InitialValueText = initialValueSpan;
			rememberToInitializeMemberInitialValues ??= new Dictionary<Member, string>();
			rememberToInitializeMemberInitialValues.Add(member, initialValueSpan);
		}
		if (!constraintsSpan.IsEmpty)
		{
			pendingConstraints ??= new List<(Member, string[])>();
			pendingConstraints.Add((member,
				constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries)));
		}
		return member;
	}

	private List<(Member member, string[] constraintsText)>? pendingConstraints;

	public void ParseDeferredConstraints(ExpressionParser parser)
	{
		if (pendingConstraints == null && rememberToInitializeMemberInitialValues == null)
			return;
		if (pendingConstraints != null)
		{
			foreach (var (member, constraintsText) in pendingConstraints)
				member.ParseConstraints(parser, constraintsText);
			pendingConstraints = null;
		}
		if (rememberToInitializeMemberInitialValues != null)
			TryInitializeMemberInitialValues(parser);
	}

	private Type? ExtractConstraintsSpanAndValueType(ExpressionParser parser,
		ReadOnlySpan<char> remainingLine, string nameAndType,
		out ReadOnlySpan<char> constraintsSpan, out string initialValueSpan)
	{
		var constraintKeywordLength = GetConstraintKeywordLength(remainingLine, nameAndType.Length);
		var equalIndex = FindStandaloneEqualIndex(remainingLine);
		if (equalIndex > 0)
		{
			constraintsSpan = remainingLine[(nameAndType.Length + 1 + constraintKeywordLength + 1)..(equalIndex - 1)];
			initialValueSpan = remainingLine[(equalIndex + 2)..].ToString();
			return GetInitialValueType(parser, nameAndType, initialValueSpan);
		}
		constraintsSpan = remainingLine[(nameAndType.Length + 1 + constraintKeywordLength + 1)..];
		initialValueSpan = "";
		return null;
	}

	private static int GetConstraintKeywordLength(ReadOnlySpan<char> remainingLine, int nameLength)
	{
		var afterName = remainingLine[(nameLength + 1)..];
		if (afterName.StartsWith(Keyword.Where))
			return Keyword.Where.Length;
		return Keyword.With.Length;
	}

	private static int FindStandaloneEqualIndex(ReadOnlySpan<char> line)
	{
		for (var index = 1; index < line.Length; index++)
			if (line[index] == EqualCharacter && line[index - 1] != '>' && line[index - 1] != '<' &&
				line[index - 1] != '!' && (index + 1 >= line.Length || line[index + 1] != '='))
				return index;
		return -1;
	}

	private const char EqualCharacter = '=';

	internal Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan, int typeLineNumber) =>
		parser.ParseExpression(
			new Body(new Method(type, typeLineNumber, parser, [nameof(GetMemberExpression)])),
			GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(memberName, remainingTextSpan));

	private ReadOnlySpan<char> GetFromConstructorCallFromUpcastableMemberOrJustEvaluate(
		string memberName, ReadOnlySpan<char> remainingTextSpan)
	{
		var memberNameWithFirstLetterCaps = memberName.MakeFirstLetterUppercase();
		return type.FindType(memberNameWithFirstLetterCaps) != null &&
			!remainingTextSpan.StartsWith(memberNameWithFirstLetterCaps)
				? string.Concat(memberNameWithFirstLetterCaps, "(", remainingTextSpan, ")").AsSpan()
				: remainingTextSpan.StartsWith(type.Name) && !char.IsUpper(memberName[0])
					? throw new CurrentTypeCannotBeInstantiatedAsMemberType(type, LineNumber,
						remainingTextSpan.ToString())
					: remainingTextSpan;
	}

	public sealed class CurrentTypeCannotBeInstantiatedAsMemberType(Type type,
		int lineNumber, string typeName) : ParsingFailed(type, lineNumber, typeName);

	private static string GetMemberType(SpanSplitEnumerator nameAndExpression)
	{
		var memberType = nameAndExpression.Current.ToString();
		while (memberType.Contains('(') && !memberType.Contains(')'))
		{
			nameAndExpression.MoveNext();
			memberType += " " + nameAndExpression.Current.ToString();
		}
		return memberType;
	}

	private static bool
		HasConstraints(string wordAfterName, ref SpanSplitEnumerator nameAndExpression) =>
		wordAfterName is Keyword.With or Keyword.Where || nameAndExpression.MoveNext() &&
		nameAndExpression.Current.ToString() is Keyword.With or Keyword.Where;

	public sealed class MemberMissingConstraintExpression(Type type, int lineNumber,
		string memberName) : ParsingFailed(type, lineNumber, memberName);

	public sealed class RedundantExplicitMemberTypeName(Type type, int lineNumber,
		string memberName, string typeName) : ParsingFailed(type, lineNumber,
		$"Member '{memberName}' already infers type '{typeName}' from its name, remove the " +
		"redundant explicit type");

	private static bool
		IsMemberTypeAny(string nameAndType, SpanSplitEnumerator nameAndExpression) =>
		nameAndType == Type.AnyLowercase ||
		nameAndExpression.Current.Equals(Type.Any, StringComparison.Ordinal);

	public sealed class MemberWithTypeAnyIsNotAllowed(Type type, int lineNumber, string name)
		: ParsingFailed(type, lineNumber, name);

	public sealed class MembersMustComeBeforeMethods(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber, line);

	public sealed class DuplicateMembersAreNotAllowed(Type type, int lineNumber, string name)
		: ParsingFailed(type, lineNumber, name);

	private string[] GetAllMethodLines(int methodFirstLineNumber)
	{
		if (type.IsTrait && IsNextLineValidMethodBody())
			throw new Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(type);
		if (!type.IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTrait(type, lines[LineNumber], LineNumber);
		IncrementLineNumberTillMethodEnd();
		return listStartLineNumber != -1
			? throw new UnterminatedMultiLineListFound(type, listStartLineNumber - 1,
				lines[listStartLineNumber])
			: lines[methodFirstLineNumber..(LineNumber + 1)];
	}

	private bool IsNextLineValidMethodBody()
	{
		if (LineNumber + 1 >= lines.Length)
			return false;
		var line = lines[LineNumber + 1];
		ValidateNestingAndLineCharacterCountLimit(line);
		if (line.StartsWith('\t'))
			return true;
		return line.Length != line.TrimStart().Length
			? throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line)
			: false;
	}

	private void ValidateNestingAndLineCharacterCountLimit(string line)
	{
		if (line.StartsWith(SixTabs, StringComparison.Ordinal))
			throw new NestingMoreThanFiveLevelsIsNotAllowed(type, LineNumber + 1);
		if (line.Length > Limit.CharacterCount)
			throw new CharacterCountMustBeWithinLimit(type, line.Length, LineNumber + 1);
	}

	private const string SixTabs = "\t\t\t\t\t\t";

	public sealed class NestingMoreThanFiveLevelsIsNotAllowed(Type type, int lineNumber)
		: ParsingFailed(type, lineNumber,
			$"Type {type.Name} has more than {Limit.NestingLevel} levels of nesting in line: " +
			$"{lineNumber + 1}");

	public sealed class CharacterCountMustBeWithinLimit(Type type, int lineLength, int lineNumber)
		: ParsingFailed(type, lineNumber,
			$"Type {
				type.Name
			} has character count {
				lineLength
			} in line: {
				lineNumber + 1
			} but limit is " + $"{Limit.CharacterCount}");

	public sealed class MethodMustBeImplementedInNonTrait(Type type, string definitionLine,
		int lineNumber) : ParsingFailed(type, lineNumber, definitionLine);

	private void IncrementLineNumberTillMethodEnd()
	{
		while (IsNextLineValidMethodBody())
		{
			LineNumber++;
			if (lines[LineNumber - 1].EndsWith(','))
				MergeMultiLineListIntoSingleLine(',');
			else if (lines[LineNumber - 1].EndsWith('+'))
				MergeMultiLineListIntoSingleLine('+');
			if (listStartLineNumber != -1 && listEndLineNumber != -1)
				SetNewLinesAndLineNumbersAfterMerge();
		}
	}

	private void MergeMultiLineListIntoSingleLine(char endCharacter)
	{
		if (listStartLineNumber == -1)
			listStartLineNumber = LineNumber - 1;
		lines[listStartLineNumber] += ' ' + lines[LineNumber].TrimStart();
		if (lines[LineNumber].EndsWith(endCharacter))
			return;
		listEndLineNumber = LineNumber;
		if (lines[listStartLineNumber].Length < Limit.MultiLineCharacterCount)
			throw new MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(type,
				listStartLineNumber - 1, lines[listStartLineNumber].Length);
	}

	private int listStartLineNumber = -1;
	private int listEndLineNumber = -1;

	public sealed class MultiLineExpressionsAllowedOnlyWhenLengthIsMoreThanHundred(Type type,
		int lineNumber, int length) : ParsingFailed(type, lineNumber, "Current length: " + length +
		$", Minimum Length for Multi line expressions: {Limit.MultiLineCharacterCount}");

	private void SetNewLinesAndLineNumbersAfterMerge()
	{
		var newLines = new List<string>(lines[..(listStartLineNumber + 1)]);
		newLines.AddRange(lines[(listEndLineNumber + 1)..]);
		lines = newLines.ToArray();
		LineNumber = listStartLineNumber;
		listStartLineNumber = -1;
		listEndLineNumber = -1;
	}

	public sealed class UnterminatedMultiLineListFound(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber, line);
}