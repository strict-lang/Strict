using System.Collections.Concurrent;
using System.Text.RegularExpressions;
#if DEBUG
using System.Runtime.CompilerServices;
#endif

namespace Strict.Language;

public partial class Type : Context, IDisposable
{
	/// <summary>
	/// Every private member and every declared variable must be used, dummies only satisfying
	/// "types without members must be traits" are forbidden. Checked when loading files.
	/// </summary>
	public void ValidateMembersAndVariablesAreUsed()
	{
		foreach (var method in methods)
			ValidateVariablesAreUsed(method);
		if (IsDataType || IsTrait || IsSingleMemberValueType)
			return;
		foreach (var member in members)
			if (!IsReservedMemberName(member.Name) && !member.IsPublic &&
				CountMemberUsage(member.Name) < 2)
				throw new UnusedMemberMustBeRemoved(this, member.Name);
	}

	private void ValidateVariablesAreUsed(Method method)
	{
		for (var index = 1; index < method.lines.Count; index++)
		{
			var declaration = DeclarationPattern.Match(method.lines[index]);
			if (!declaration.Success)
				continue;
			var usage = new Regex(@"\b" + declaration.Groups[1].Value + @"\b");
			if (!method.lines.Where((line, lineIndex) => lineIndex != index).Any(usage.IsMatch))
				throw new UnusedMethodVariableMustBeRemoved(this, declaration.Groups[1].Value);
		}
	}

	public sealed class UnusedMethodVariableMustBeRemoved(Type type, string name)
		: ParsingFailed(type, 0, name);

	/// <summary>
	/// Wrappers like Degrees with a single "has number" use that member through value or from.
	/// </summary>
	private bool IsSingleMemberValueType =>
		members.Count(member => !member.IsConstant) == 1 && (CountMemberUsage(ValueLowercase) > 0 ||
			methods.Any(method => method.Name == Method.From));

	private static bool IsReservedMemberName(string name) =>
		name is ValueLowercase or IteratorLowercase or ElementsLowercase or GenericLowercase;

	public sealed class UnusedMemberMustBeRemoved(Type type, string memberName)
		: ParsingFailed(type, 0, memberName);

	public int CountMemberUsage(string memberName) =>
		Lines.Count(line => line.Contains(' ' + memberName) || line.Contains('\t' + memberName) ||
			line.Contains('(' + memberName));
}
