using System.Diagnostics;
using System.Runtime.CompilerServices;
using static Strict.Language.NamedType;

namespace Strict.Language;

/// <summary>
/// Keeps all known types for use, if in <see cref="Package"/> contains all known types and traits
/// the context is inside a type, all members are available as well, in a method more information
/// is available. The high-level context knows all types, low-level scope in methods is managed via
/// <see cref="Body"/> (which is every MethodBody, If.Then, If.Else, or For).
/// </summary>
[DebuggerDisplay("Property={FullName}")]
public abstract class Context
{
	protected Context(Context? parent, string name
#if DEBUG
		, string callerFilePath, int callerLineNumber, string callerMemberName
#endif
	)
	{
		var isNotGeneric = this is not GenericType && this is not GenericTypeImplementation;
		if (isNotGeneric && parent != null && (string.IsNullOrWhiteSpace(name) ||
			IsNotMethodOrPackageAndNotConflictingType(this, parent, name)))
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		if (this is Package && !name.IsAlphaNumericWithAllowedSpecialCharacters())
			throw new PackageNameMustBeAWordWithoutSpecialCharacters(name);
		if (isNotGeneric && !string.IsNullOrEmpty(name) && !name.Length.IsWithinLimit() &&
			!name.IsOperatorOrAllowedMethodName())
			throw new NameLengthIsNotWithinTheAllowedLimit(name);
		Parent = parent!;
		Name = name;
#if DEBUG
		this.callerFilePath = callerFilePath;
		this.callerLineNumber = callerLineNumber;
		this.callerMemberName = callerMemberName;
#endif
		FullName = string.IsNullOrEmpty(parent?.Name) || parent.Name is nameof(Base)
			? name
			: parent + "." + name;
	}

#if DEBUG
	protected readonly string callerFilePath;
	protected readonly int callerLineNumber;
	protected readonly string callerMemberName;
#endif

	private static bool IsNotMethodOrPackageAndNotConflictingType(Context context, Context parent,
		string name)
	{
		var lastLetterNumber = -1;
		return context is not Method && context is not Package &&
			!name.IsWordOrWordWithNumberAtEnd(out lastLetterNumber) ||
			lastLetterNumber != -1 && parent.FindType(name[..^1]) != null;
	}

	public sealed class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name)
		: Exception(name);

	public sealed class PackageNameMustBeAWordWithoutSpecialCharacters(string name) : Exception(
		"Name " + name +
		" ;Allowed characters: Alphabets, Numbers or '-' in the middle or end of the name");

	public Context Parent { get; }
	public string Name { get; }
	public string FullName { get; }

	// ReSharper disable once InconsistentlySynchronizedField
	public Type GetType(string name) =>
		TryGetType(name) ?? throw new TypeNotFound(name, FullName, types.Keys.ToWordList());

	internal Type? TryGetType(string name)
	{
		lock (types)
		{
			if (types.TryGetValue(name, out var type))
				return type;
			var result = GuessTypeFromName();
			types[name] = result;
			return result;
		}

		Type? GuessTypeFromName()
		{
			if (name == Name || this is Type && ((Type)this).IsGeneric &&
				name.StartsWith(Name, StringComparison.Ordinal) &&
				name == Name + GenericImplementationPostfix)
				return (Type)this;
			if (name.StartsWith("List", StringComparison.Ordinal) && name.Length > 4 && name[4] != '(')
				throw new ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(name);
			if (name.EndsWith('s'))
				return TryGetTypeFromPluralNameAsListWithSingularName(name);
			if (name.EndsWith(')') && name.Contains('('))
				return GetGenericTypeWithArguments(name);
			return FindFullType(name) ?? FindType(name, this);
		}
	}

	private readonly IDictionary<string, Type?> types = new Dictionary<string, Type?>();

	public sealed class ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(string typeName)
		: Exception($"List should not be used as prefix for {
			typeName
		} instead use {
			typeName.Replace("List", "").GetTextInsideBrackets().Pluralize()
		}");

	/// <summary>
	/// Always convert a plural name into List(SingularName), e.g., Texts becomes List(Text)
	/// </summary>
	private Type? TryGetTypeFromPluralNameAsListWithSingularName(string name)
	{
		var singularName = name[..^1];
		if (singularName == Base.Generic)
			return GetType(Base.List);
		var elementType = FindFullType(singularName) ?? FindType(singularName, this);
		if (elementType != null)
			return GetListImplementationType(elementType);
		return FindFullType(name) ?? FindType(name, this);
	}

	private const string GenericImplementationPostfix = "(" + Base.Generic + ")";

	private Type GetGenericTypeWithArguments(string name)
	{
		var mainType = GetType(name[..name.IndexOf('(')]);
		var rest = name[(mainType.Name.Length + 1)..^1];
		var arguments = rest.Split(',', StringSplitOptions.TrimEntries);
		if (rest.Contains("Generic"))
		{
			var namedTypes = GetNamedTypes(mainType, arguments);
			return mainType.Package.FindDirectType(mainType.GetImplementationName(namedTypes)) ??
				new GenericType(mainType, namedTypes);
		}
		var argumentTypes = GetArgumentTypes(arguments);
		return mainType.GetGenericImplementation(argumentTypes);
	}

	private static NamedType[] GetNamedTypes(Type mainType, IReadOnlyList<string> argumentTypeNames)
	{
		var namedTypes = new NamedType[argumentTypeNames.Count];
		for (var index = 0; index < argumentTypeNames.Count; index++)
			namedTypes[index] = new Parameter(mainType, argumentTypeNames[index]);
		return namedTypes;
	}

	private Type[] GetArgumentTypes(IReadOnlyList<string> argumentTypeNames)
	{
		var argumentTypes = new Type[argumentTypeNames.Count];
		for (var index = 0; index < argumentTypeNames.Count; index++)
			argumentTypes[index] = GetType(argumentTypeNames[index]);
		return argumentTypes;
	}

	public sealed class TypeArgumentsCountDoesNotMatchGenericType(Type mainType,
		IReadOnlyCollection<Type> typeArguments) : Exception("The generic type " + mainType +
		" needs these type arguments: " + mainType.GetGenericTypeArguments().ToBrackets() +
		", this does not match provided types: " + typeArguments.ToBrackets());

	public GenericTypeImplementation GetListImplementationType(Type implementation) =>
		GetType(Base.List).GetGenericImplementation(implementation);

	public GenericTypeImplementation GetDictionaryImplementationType(Type keyType, Type valueType) =>
		GetType(Base.Dictionary).GetGenericImplementation(keyType, valueType);

	private Type? FindFullType(string name) =>
		name.Contains('.')
			? name == FullName
				? this as Type
				: GetPackage()?.FindFullType(name)
			: null;

	public override string ToString() => FullName;

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	public Package? GetPackage() =>
		this is Package package
			// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
			? Parent is null
				? null
				: package
			: Parent.GetPackage();

	public sealed class TypeNotFound(string typeName, string contextFullName, string contextTypes)
		: Exception($"{typeName} not found in {contextFullName}, available types: " + contextTypes);

	public abstract Type? FindType(string name, Context? searchingFrom = null);
}