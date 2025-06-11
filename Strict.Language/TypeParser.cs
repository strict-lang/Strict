using System.Data;

namespace Strict.Language;

public sealed class TypeParser(Type type, string[] lines)
{
	private string[] lines = lines;

	public void ParseMembersAndMethods(ExpressionParser parser)
	{
		for (; LineNumber < lines.Length; LineNumber++)
			TryParse(parser, LineNumber);
	}

	public int LineNumber { get; private set; }

	private void TryParse(ExpressionParser parser, int rememberStartMethodLineNumber)
	{
		try
		{
			ParseLineForMembersAndMethods(parser);
		}
		catch (Context.TypeNotFound ex)
		{
			throw new ParsingFailed(type, rememberStartMethodLineNumber, ex.Message, ex);
		}
		catch (ParsingFailed)
		{
			throw;
		}
		catch (Exception ex)
		{
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
			type.Members.Add(GetNewMember(parser, true));
		else
			type.Methods.Add(new Method(type, LineNumber, parser, GetAllMethodLines()));
	}

	private string ValidateCurrentLineIsNonEmptyAndTrimmed()
	{
		var line = lines[LineNumber];
		if (line.Length == 0)
			throw new EmptyLineIsNotAllowed(type, LineNumber);
		if (char.IsWhiteSpace(line[0]))
			throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line);
		if (char.IsWhiteSpace(line[^1]))
			throw new ExtraWhitespacesFoundAtEndOfLine(type, LineNumber, line);
		return line;
	}

	public sealed class EmptyLineIsNotAllowed(Type type, int lineNumber)
		: ParsingFailed(type, lineNumber);

	public sealed class ExtraWhitespacesFoundAtBeginningOfLine(Type type, int lineNumber,
		string message, string method = "") : ParsingFailed(type, lineNumber, message, method);

	public sealed class ExtraWhitespacesFoundAtEndOfLine(Type type, int lineNumber,
		string message, string method = "") : ParsingFailed(type, lineNumber, message, method);

	private Member GetNewMember(ExpressionParser parser, bool usedMutableKeyword = false)
	{
		var member = ParseMember(parser, lines[LineNumber].AsSpan((usedMutableKeyword
			? Type.MutableWithSpaceAtEnd
			: Type.HasWithSpaceAtEnd).Length), usedMutableKeyword);
		if (type.Members.Any(m => m.Name == member.Name))
			throw new DuplicateMembersAreNotAllowed(type, LineNumber, member.Name);
		return member;
	}

	private Member ParseMember(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		bool usedMutableKeyword)
	{
		if (type.Methods.Count > 0)
			throw new MembersMustComeBeforeMethods(type, LineNumber, remainingLine.ToString());
		try
		{
			return TryParseMember(parser, remainingLine, usedMutableKeyword);
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
		bool usedMutableKeyword)
	{
		var nameAndExpression = remainingLine.Split();
		nameAndExpression.MoveNext();
		var nameAndType = nameAndExpression.Current.ToString();
		if (nameAndExpression.MoveNext())
		{
			var wordAfterName = nameAndExpression.Current.ToString();
			if (nameAndExpression.Current[0] == EqualCharacter)
				return new Member(type, nameAndType,
					GetMemberExpression(parser, nameAndType,
						remainingLine[(nameAndType.Length + 3)..]), usedMutableKeyword);
			if (wordAfterName != Keyword.With)
				nameAndType += " " + GetMemberType(nameAndExpression);
			if (HasConstraints(wordAfterName, ref nameAndExpression))
				return !nameAndExpression.MoveNext()
					? throw new MemberMissingConstraintExpression(type, LineNumber, nameAndType)
					: IsMemberTypeAny(nameAndType, nameAndExpression)
						? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
						: GetMemberWithConstraints(parser, remainingLine, usedMutableKeyword, nameAndType);
			if (nameAndExpression.Current[0] == EqualCharacter)
				throw new NamedType.AssignmentWithInitializerTypeShouldNotHaveNameWithType(nameAndType);
		}
		return IsMemberTypeAny(nameAndType, nameAndExpression)
			? throw new MemberWithTypeAnyIsNotAllowed(type, LineNumber, nameAndType)
			: new Member(type, nameAndType, null, usedMutableKeyword);
	}

	private Member GetMemberWithConstraints(ExpressionParser parser, ReadOnlySpan<char> remainingLine,
		bool usedMutableKeyword, string nameAndType)
	{
		var member = new Member(type, nameAndType,
			ExtractConstraintsSpanAndValueExpression(parser, remainingLine, nameAndType,
				out var constraintsSpan), usedMutableKeyword);
		if (!constraintsSpan.IsEmpty)
			member.ParseConstraints(parser,
				constraintsSpan.ToString().Split(BinaryOperator.And, StringSplitOptions.TrimEntries));
		return member;
	}

	private Expression? ExtractConstraintsSpanAndValueExpression(ExpressionParser parser,
		ReadOnlySpan<char> remainingLine, string nameAndType,
		out ReadOnlySpan<char> constraintsSpan)
	{
		var equalIndex = remainingLine.IndexOf(EqualCharacter);
		if (equalIndex > 0)
		{
			constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..(equalIndex - 1)];
			return GetMemberExpression(parser, nameAndType,
				remainingLine[(equalIndex + 2)..]);
		}
		constraintsSpan = remainingLine[(nameAndType.Length + 1 + Keyword.With.Length + 1)..];
		return null;
	}

	private const char EqualCharacter = '=';

	internal Expression GetMemberExpression(ExpressionParser parser, string memberName,
		ReadOnlySpan<char> remainingTextSpan) =>
		parser.ParseExpression(new Body(new Method(type, 0, parser, [Type.EmptyBody])),
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
		if (line.Length != line.TrimStart().Length)
			throw new ExtraWhitespacesFoundAtBeginningOfLine(type, LineNumber, line);
		return false;
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
		int lineNumber, int length) : ParsingFailed(type, lineNumber,
		"Current length: " + length + $", Minimum Length for Multi line expressions: {
			Limit.MultiLineCharacterCount
		}");

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