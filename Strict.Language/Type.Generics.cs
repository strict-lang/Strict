using System.Collections.Concurrent;
using System.Text.RegularExpressions;
#if DEBUG
using System.Runtime.CompilerServices;
#endif

namespace Strict.Language;

public partial class Type : Context, IDisposable
{
	private bool OneOfFirstThreeLinesContainsGeneric()
	{
		for (var line = 0; line < Lines.Length && line < 3; line++)
			if (HasGenericMember(Lines[line]) || (HasGenericMethodHeader(Lines[line]) &&
				line + 1 < Lines.Length && !Lines[line + 1].StartsWith('\t')))
				return true;
		return false;
	}

	private static bool HasGenericMember(string line) =>
		(line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal) ||
			line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal)) &&
		(line.Contains(GenericUppercase, StringComparison.Ordinal) ||
			line.Contains(GenericLowercase, StringComparison.Ordinal)) &&
		!IsNamedMemberWithAlternativeOrDefault(line);

	// "has Name Generic or None" or "has Name Generic = default" have an alternative/default so the
	// type is not parameterized by the generic member and should not be treated as IsGeneric.
	// "has Generic" (unnamed) and "has Name Generic" (no alternative) still make the type generic.
	private static bool IsNamedMemberWithAlternativeOrDefault(string line)
	{
		var parts = line.Split(' ');
		return parts.Length > 3 && parts[2] is GenericUppercase or GenericLowercase &&
			!line.Contains('(');
	}

	private static bool HasGenericMethodHeader(string line) =>
		!line.StartsWith(HasWithSpaceAtEnd, StringComparison.Ordinal) &&
		!line.StartsWith(MutableWithSpaceAtEnd, StringComparison.Ordinal) &&
		(line.Contains(GenericUppercase, StringComparison.Ordinal) ||
			line.Contains(GenericLowercase, StringComparison.Ordinal));

	public GenericTypeImplementation GetGenericImplementation(params Type[] implementationTypes)
	{
		var key = GetImplementationName(implementationTypes);
		lock (genericImplementationLock)
		{
			return GetGenericImplementation(key) ?? CreateGenericImplementation(key, implementationTypes);
		}
	}

	internal string GetImplementationName(Type[] implementationTypes)
	{
		var key = "";
		for (var i = 0; i < implementationTypes.Length; i++)
			key += (key == ""
				? ""
				: ", ") + implementationTypes[i].Name;
		return Name + "(" + key + ")";
	}

	internal string GetImplementationName(IReadOnlyList<NamedType> implementationTypes)
	{
		var key = "";
		for (var i = 0; i < implementationTypes.Count; i++)
			key += (key == ""
				? ""
				: ", ") + implementationTypes[i];
		return Name + "(" + key + ")";
	}

	private GenericTypeImplementation? GetGenericImplementation(string key)
	{
		if (!IsGeneric)
			throw new CannotGetGenericImplementationOnNonGeneric(Name, key);
		cachedGenericTypes ??=
			new Dictionary<string, GenericTypeImplementation>(StringComparer.Ordinal);
		return cachedGenericTypes.GetValueOrDefault(key);
	}

	/// <summary>
	/// Most often called for List (or the Iterator trait), which we want to optimize for
	/// </summary>
	private GenericTypeImplementation CreateGenericImplementation(string key,
		Type[] implementationTypes)
	{
		if (((IsList || IsIterator || IsMutable) && implementationTypes.Length == 1) ||
			GetGenericTypeArguments().Count == implementationTypes.Length ||
			HasMatchingConstructor(implementationTypes))
		{
			var genericType = new GenericTypeImplementation(this, implementationTypes, key);
			cachedGenericTypes!.Add(key, genericType);
			return genericType;
		}
		throw new TypeArgumentsCountDoesNotMatchGenericType(this, implementationTypes);
	}

	private bool HasMatchingConstructor(Type[] implementationTypes) =>
		typeMethodFinder.FindFromMethodImplementation(implementationTypes) != null;

	public sealed class CannotGetGenericImplementationOnNonGeneric(string name, string key)
		: Exception("Type: " + name + ", Generic Implementation: " + key);

	[Log]
	public HashSet<NamedType> GetGenericTypeArguments()
	{
		if (!IsGeneric)
			throw new TypeMustBeGenericToCallThis(this); //ncrunch: no coverage
		var genericArguments = new HashSet<NamedType>();
		foreach (var member in Members)
			if (member.Type is GenericType genericType)
				foreach (var namedType in genericType.GenericImplementations)
					genericArguments.Add(namedType);
			else if (member.Type.IsList || member.Type.IsIterator)
				genericArguments.Add(new Parameter(this, GenericUppercase));
			else if (member.Type.IsGeneric)
				genericArguments.Add(member);
		return genericArguments.Count == 0
			? throw new InvalidGenericTypeWithoutGenericArguments(this)
			: genericArguments;
	}

	//ncrunch: no coverage start
	public sealed class TypeMustBeGenericToCallThis(Type type) : Exception(type.FullName);

	public sealed class InvalidGenericTypeWithoutGenericArguments(Type type) : Exception(
		"This type is broken and needs to be fixed, check the creation: " + type + ", Package: " +
		type.Package + ", file=" + type.FilePath);
	//ncrunch: no coverage end

	public Type GetFirstImplementation() => ((GenericTypeImplementation)this).ImplementationTypes[0];
}
