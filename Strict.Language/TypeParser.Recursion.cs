using System.Globalization;

namespace Strict.Language;

public sealed partial class TypeParser
{
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
			foreach (var param in inside.Split(',',
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
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

	private void SearchForMethodCalls(string methodName, string signature, string line,
		IReadOnlyList<string> paramNames)
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
					throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber, signature, argText,
						line.Trim());
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

	private void SearchForMemberMethodCalls(string methodName, string signature, string line,
		IReadOnlyList<string> paramNames)
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

	private void CheckRecursionCallingThisMethod(string signature, string line,
		IReadOnlyList<string> paramNames, int dotIdx, string dotPattern)
	{
		var argsStart = dotIdx + dotPattern.Length - 1;
		var argsEnd = line.IndexOf(')', argsStart + 1);
		if (argsEnd > argsStart)
		{
			var argText = line[(argsStart + 1)..argsEnd];
			if (AreParametersEqual(argText, paramNames))
				throw new SelfRecursiveCallWithSameArgumentsDetected(type, LineNumber, signature, argText,
					line.Trim());
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
			if (endArgs < 0)
				continue;
			var args = line[startArgs..endArgs].Split(',',
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);
			if (args.Length == 2 &&
				long.TryParse(args[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var start) &&
				long.TryParse(args[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var end))
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

	public sealed class SelfRecursiveCallWithSameArgumentsDetected(Type type,
		int lineNumber,
		string signature,
		string argumentNames,
		string line) : ParsingFailed(type, lineNumber,
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
						parameter.Split(' ', StringSplitOptions.RemoveEmptyEntries).Skip(1).FirstOrDefault() ??
						Type.Any));
	}

	public sealed class HugeConstantRangeNotAllowed(Type type,
		int lineNumber,
		string line,
		long span,
		long limit)
		: ParsingFailed(type, lineNumber, $"Range size {span} exceeds limit {limit}: " + line);

	private void DetectRedundantReturn(IReadOnlyList<string> checkLines, Method method)
	{
		if (checkLines[^1].StartsWith("\treturn ", StringComparison.Ordinal))
			throw new Body.ReturnAsLastExpressionIsNotNeeded(new Body(method));
		if (checkLines.Count < 3)
			return;

		//TODO: way to complicated and slow just to check for /t at the beginning of a line
		static int GetIndent(string line) => line.TakeWhile(c => c == '\t').Count();
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

	public sealed class RedundantReturnPreviousLineContainsValueAlready(Type type,
		int lineNumber,
		string prevLine,
		string variableName) : ParsingFailed(type, lineNumber, prevLine, variableName);
}
