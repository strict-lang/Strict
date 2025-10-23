using System.Globalization;

namespace Strict.Language;

public sealed class TypeParser(Type type, string[] lines)
{
	private string[] lines = lines;

	public void ParseMembersAndMethods(ExpressionParser parser)
	{
		for (LineNumber = 0; LineNumber < lines.Length; LineNumber++)
			TryParse(parser, LineNumber);
		if (rememberToInitializeMemberInitialValues != null)
			TryInitializeMemberInitialValues(parser);
	}

	private void TryInitializeMemberInitialValues(ExpressionParser parser)
	{
		try
		{
			foreach (var pair in rememberToInitializeMemberInitialValues!)
				pair.Key.InitialValue = GetMemberExpression(parser, pair.Key.Name, pair.Value);
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

	internal int LineNumber = -1; //property is slower, especially in debug: { get; private set; }

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
			var methodLines = GetAllMethodLines();
			DetectTrivialEndlessRecursionInFrom(methodLines);
			DetectSelfRecursionWithSameArguments(methodLines);
			DetectHugeConstantRange(methodLines);
			type.Methods.Add(new Method(type, LineNumber, parser, methodLines));
		}
	}

	/// <summary>
	/// If a from(...) method contains a same-type constructor call like TypeName(constant) and the
	/// call's argument does not reference any parameter, it will just recursively call itself
	/// forever (e.g., Character.from used Character(0), which would forever call itself).
	/// </summary>
	private void DetectTrivialEndlessRecursionInFrom(IReadOnlyList<string> methodLines)
	{
		if (methodLines.Count == 0)
			return;
		var signature = methodLines[0];
		var openParen = signature.IndexOf('(');
		if (openParen <= 0)
			return;
		var methodName = signature[..openParen];
		if (!methodName.Equals("from", StringComparison.Ordinal))
			return;
		var paramNames = CollectParameterNamesFromSignature(signature, openParen);
		// Inspect body lines (all subsequent lines), skip inline tests (contain " is ")
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
				continue;
			var argText = line[startArgs..endArgs];
			// If argText does not contain any parameter name, it's a constant/self call -> flag
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
				// param format is "name Type" or just "name"
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
		if (methodLines.Count == 0 || type.Name == Base.System)
			return;
		var signature = methodLines[0];
		var openParen = signature.IndexOf('(');
		var closeParen = signature.IndexOf(')', openParen + 1);
		if (openParen <= 0 || closeParen <= openParen)
			return;
		var methodName = signature[..openParen].Trim();
		// Extract parameter names from signature in order
		var paramNames = new List<string>();
		var inside = signature[(openParen + 1)..closeParen];
		foreach (var param in inside.Split(',',
			StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
		{
			var parts = param.Split(' ', StringSplitOptions.RemoveEmptyEntries);
			if (parts.Length > 0)
				paramNames.Add(parts[0]);
		}
		if (paramNames.Count == 0)
			return;

		bool ArgsEqualParams(string argText)
		{
			var argNames = argText.Split(',',
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
			if (argNames.Length != paramNames.Count)
				return false;
			for (var i = 0; i < argNames.Length; i++)
				if (!argNames[i].Equals(paramNames[i], StringComparison.Ordinal))
					return false;
			return true;
		}

		for (var i = 1; i < methodLines.Count; i++)
		{
			var line = methodLines[i];
			if (IsNonTestMethodLine(line))
				continue;
			// 1) Dot calls: receiver.Method(...)
			var searchStart = 0;
			var dotPattern = "." + methodName + "(";
			while (true)
			{
				var dotIdx = line.IndexOf(dotPattern, searchStart, StringComparison.Ordinal);
				if (dotIdx < 0)
					break;
				// Extract receiver token before the dot
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
				{
					var argsStart = dotIdx + dotPattern.Length - 1; // position at '('
					var argsEnd = line.IndexOf(')', argsStart + 1);
					if (argsEnd > argsStart)
					{
						var argText = line[(argsStart + 1)..argsEnd];
						if (ArgsEqualParams(argText))
							throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber,
								line.Trim());
					}
				}
				searchStart = dotIdx + dotPattern.Length;
			}
			// 2) Direct calls: Method(...)
			searchStart = 0;
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
					if (ArgsEqualParams(argText))
						throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber,
							line.Trim());
				}
				searchStart = directIdx + directPattern.Length;
			}
		}
	}

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
			if (args.Length != 2)
				continue;
			if (long.TryParse(args[0], NumberStyles.Integer, CultureInfo.InvariantCulture,
					out var start) && long.TryParse(args[1], NumberStyles.Integer,
					CultureInfo.InvariantCulture, out var end))
			{
				var span = Math.Abs(end - start);
				if (span > MaximumRangeAllowed)
					throw new HugeConstantRangeNotAllowed(type, LineNumber, line.Trim(), span,
						MaximumRangeAllowed);
			}
		}
	}

	public sealed class TrivialEndlessSelfConstructionDetected(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber,
			"Endless recursion via self-constructor call in from: " + line);

	public sealed class SelfRecursiveCallWithSameArgumentsDetected(Type type, int lineNumber,
		string line) : ParsingFailed(type, lineNumber,
		"Self-recursive call with same arguments detected: " + line);

	public sealed class HugeConstantRangeNotAllowed(Type type, int lineNumber, string line,
		long span, long limit) : ParsingFailed(type, lineNumber,
		$"Range size {span} exceeds limit {limit}: " + line);

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
			throw new ParsingFailed(type, LineNumber, ex.Message.Split('\n').Take(2).ToWordList("\n"), ex);
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
				if (usedKeyword == Keyword.Constant &&
					int.TryParse(constantValue, out var forcedEnumNumber))
					type.AutogeneratedEnumValue = forcedEnumNumber;
				var member = new Member(type, nameAndType,
					GetInitialValueType(parser, nameAndType, constantValue), usedKeyword);
				rememberToInitializeMemberInitialValues ??= new Dictionary<Member, string>();
				rememberToInitializeMemberInitialValues.Add(member, constantValue.ToString());
				return member;
			}
			if (wordAfterName != Keyword.With)
				nameAndType += " " + GetMemberType(nameAndExpression);
			if (HasConstraints(wordAfterName, ref nameAndExpression))
				return !nameAndExpression.MoveNext()
					? throw new MemberMissingConstraintExpression(type, LineNumber, nameAndType)
					: IsMemberTypeAny(nameAndType, nameAndExpression)
						? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
						: GetMemberWithConstraints(parser, remainingLine, usedKeyword, nameAndType);
			if (nameAndExpression.Current[0] == EqualCharacter)
				throw new NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType(nameAndType);
		}
		return IsMemberTypeAny(nameAndType, nameAndExpression)
			? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
			: usedKeyword == Keyword.Constant
				? new Member(type, nameAndType, type.GetType(Base.Number), usedKeyword)
				{
					InitialValue = GetMemberExpression(parser, nameAndType,
						(type.AutogeneratedEnumValue++).ToString())
				}
				: new Member(type, nameAndType, null, usedKeyword);
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
			return type.GetType(Base.Text);
		if (constantValue is "true" || constantValue is "false")
			return type.GetType(Base.Boolean);
		return constantValue.TryParseNumber(out _)
			? type.GetType(Base.Number)
			: GetMemberExpression(parser, nameAndType, constantValue).ReturnType;
	}

	private Dictionary<Member, string>? rememberToInitializeMemberInitialValues = null;

	private Member GetMemberWithConstraints(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		string usedKeyword, string nameAndType)
	{
		var member = new Member(type, nameAndType,
			ExtractConstraintsSpanAndValueType(parser, remainingLine, nameAndType,
				out var constraintsSpan, out var initialValueSpan), usedKeyword);
		if (initialValueSpan != "")
		{
			rememberToInitializeMemberInitialValues ??= new Dictionary<Member, string>();
			rememberToInitializeMemberInitialValues.Add(member, initialValueSpan);
		}
		if (!constraintsSpan.IsEmpty)
			member.ParseConstraints(parser,
				constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries));
		return member;
	}

	private Type? ExtractConstraintsSpanAndValueType(ExpressionParser parser,
		ReadOnlySpan<char> remainingLine, string nameAndType,
		out ReadOnlySpan<char> constraintsSpan, out string initialValueSpan)
	{
		var equalIndex = remainingLine.IndexOf(EqualCharacter);
		if (equalIndex > 0)
		{
			constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..(equalIndex - 1)];
			initialValueSpan = remainingLine[(equalIndex + 2)..].ToString();
			return GetInitialValueType(parser, nameAndType, initialValueSpan);
		}
		constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..];
		initialValueSpan = "";
		return null;
	}

	private const char EqualCharacter = '=';

	internal Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan) =>
		parser.ParseExpression(new Body(new Method(type, 0, parser, [nameof(GetMemberExpression)])),
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
		wordAfterName == Keyword.With || nameAndExpression.MoveNext() &&
		nameAndExpression.Current.ToString() == Keyword.With;

	public sealed class MemberMissingConstraintExpression(Type type, int lineNumber,
		string memberName) : ParsingFailed(type, lineNumber, memberName);

	private static bool
		IsMemberTypeAny(string nameAndType, SpanSplitEnumerator nameAndExpression) =>
		nameAndType == Base.AnyLowercase ||
		nameAndExpression.Current.Equals(Base.Any, StringComparison.Ordinal);

	public sealed class MemberWithTypeAnyIsNotAllowed(Type type, int lineNumber, string name)
		: ParsingFailed(type, lineNumber, name);

	public sealed class MembersMustComeBeforeMethods(Type type, int lineNumber, string line)
		: ParsingFailed(type, lineNumber, line);

	public sealed class DuplicateMembersAreNotAllowed(Type type, int lineNumber, string name)
		: ParsingFailed(type, lineNumber, name);

	private string[] GetAllMethodLines()
	{
		if (type.IsTrait && IsNextLineValidMethodBody())
			throw new Type.TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(type);
		if (!type.IsTrait && !IsNextLineValidMethodBody())
			throw new MethodMustBeImplementedInNonTrait(type, lines[LineNumber], LineNumber);
		var methodLineNumber = LineNumber;
		IncrementLineNumberTillMethodEnd();
		return listStartLineNumber != -1
			? throw new UnterminatedMultiLineListFound(type, listStartLineNumber - 1,
				lines[listStartLineNumber])
			: lines[methodLineNumber..(LineNumber + 1)];
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