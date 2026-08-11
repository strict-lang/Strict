using Strict.Language;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public static class PackageSynchronizationExtension
{
	public static Type SynchronizeAndGetType(this Package package, string typeName,
		IEnumerable<string> code)
	{
		var lines = code.Select(line => line.Replace("    ", "\t", StringComparison.Ordinal)).ToArray();
		var outdatedType = package.FindDirectType(typeName);
		// Keep a restore copy: parse failure must not leave the package without this type
		// (opening Boolean.strict with a bad buffer used to wipe Boolean for the whole session).
		string[]? restoreLines = outdatedType?.Lines;
		if (outdatedType != null)
			package.Remove(outdatedType);
		try
		{
			return new Type(package, new TypeLines(typeName, lines)).
				ParseMembersAndMethods(new MethodExpressionParser());
		}
		catch
		{
			if (restoreLines != null && package.FindDirectType(typeName) == null)
				try
				{
					new Type(package, new TypeLines(typeName, restoreLines)).
						ParseMembersAndMethods(new MethodExpressionParser());
				}
				catch
				{
					// Best-effort restore; rethrow the original parse error below.
				}
			throw;
		}
	}
}