using Strict.Language;
using Strict.Expressions;
using Type = Strict.Language.Type;

namespace Strict.LanguageServer;

public static class PackageSynchronizationExtension
{
	public static Type SynchronizeAndGetType(this Package package, string typeName,
		IEnumerable<string> code)
	{
		var outdatedType = package.FindDirectType(typeName);
		if (outdatedType != null)
			package.Remove(outdatedType);
		var type = new Type(package,
				new TypeLines(typeName, code.Select(line => line.Replace("    ", "\t")).ToArray())).
			ParseMembersAndMethods(new MethodExpressionParser());
		return type;
	}
}