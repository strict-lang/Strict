using System.Runtime.CompilerServices;
using static Strict.Language.NamedType;

namespace Strict.Language;

/// <summary>
/// Keeps all known types for use, if in <see cref="Package"/> contains all known types and traits
/// the context is inside a type, all members are available as well, in a method more information
/// is available. The high level context knows all types, low level scope in methods is managed via
/// <see cref="Body"/> (which is every MethodBody, If.Then, If.Else or For).
/// </summary>
public abstract class Context
{
	protected Context(Context? parent, string name)
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
		FullName = string.IsNullOrEmpty(parent?.Name) || parent.Name is nameof(Base)
			? name
			: parent + "." + name;
	}

	private static bool IsNotMethodOrPackageAndNotConflictingType(Context context, Context parent,
		string name)
	{
		var lastLetterNumber = -1;
		return context is not Method && context is not Package &&
			!name.IsWordOrWordWithNumberAtEnd(out lastLetterNumber) ||
			lastLetterNumber != -1 && parent.FindType(name[..^1]) != null;
	}

	public sealed class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers : Exception
	{
		public NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name) : base(name) { }
	}

	public sealed class PackageNameMustBeAWordWithoutSpecialCharacters : Exception
	{
		public PackageNameMustBeAWordWithoutSpecialCharacters(string name) : base("Name " + name +
			" ;Allowed characters: Alphabets, Numbers or '-' in the middle or end of the name") { }
	}

	public Context Parent { get; protected set; }
	public string Name { get; }
	public string FullName { get; }

	public Type GetType(string name)
	{
		if (cachedTypes != null && cachedTypes.TryGetValue(name, out var type))
			return type;
		cachedTypes ??= new Dictionary<string, Type>();
		var foundType = GetTypeFromPackages(name);
		cachedTypes.Add(name, foundType);
		return foundType;
	}

	private Dictionary<string, Type>? cachedTypes;

	private Type GetTypeFromPackages(string name)
	{
		if (name == Name || this is Type && ((Type)this).IsGeneric &&
			name.StartsWith(Name, StringComparison.Ordinal) &&
			name == Name + GenericImplementationPostfix)
			return (Type)this;
		if (name.EndsWith('s'))
			return GetTypeFromPluralNameAsListWithSingularName(name);
		if (name.EndsWith(')') && name.Contains('('))
			return GetGenericTypeWithArguments(name);
		return (FindFullType(name) ?? FindType(name, this)) ??
			throw new TypeNotFound(name, FullName);
	}

	/// <summary>
	/// Always convert plural name into List(SingularName), e.g. Texts becomes List(Text)
	/// </summary>
	private Type GetTypeFromPluralNameAsListWithSingularName(string name)
	{
		var singularName = name[..^1];
		if (singularName == Base.Generic)
			return GetType(Base.List);
		var elementType = FindFullType(singularName) ?? FindType(singularName, this);
		if (elementType != null)
			return GetListImplementationType(elementType);
		return (FindFullType(name) ?? FindType(name, this)) ?? throw new TypeNotFound(name, FullName);
	}

	private const string GenericImplementationPostfix = "(" + Base.Generic + ")";

	private Type GetGenericTypeWithArguments(string name)
	{
		var mainType = GetType(name[..name.IndexOf('(')]);
		var rest = name[(mainType.Name.Length + 1)..^1];
		if (rest.Contains("Generic"))
			return new GenericType(mainType,
				GetNamedTypes(mainType, rest.Split(',', StringSplitOptions.TrimEntries)));
		var argumentTypes = GetArgumentTypes(rest.Split(',', StringSplitOptions.TrimEntries));
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

	public sealed class TypeArgumentsCountDoesNotMatchGenericType : Exception
	{
		public TypeArgumentsCountDoesNotMatchGenericType(Type mainType,
			IReadOnlyCollection<Type> typeArguments) : base("The generic type " + mainType +
			" needs these type arguments: " + mainType.GetGenericTypeArguments().ToBrackets() +
			", this does not match provided types: " + typeArguments.ToBrackets()) { }
	}

	public GenericTypeImplementation GetListImplementationType(Type implementation) =>
		GetType(Base.List).GetGenericImplementation(implementation);

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

	public sealed class TypeNotFound : Exception
	{
		public TypeNotFound(string typeName, string contextFullName) : base(
			$"{typeName} not found in {contextFullName}") { }
	}

	public abstract Type? FindType(string name, Context? searchingFrom = null);
}