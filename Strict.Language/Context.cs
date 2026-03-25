using System.Diagnostics;
using System.Runtime.CompilerServices;
using static Strict.Language.NamedType;

[assembly: InternalsVisibleTo("Strict.Expressions")]

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
		if (isNotGeneric && !string.IsNullOrEmpty(name) && !name.Length.IsNameLengthWithinLimit() &&
			!name.IsOperatorOrAllowedMethodName())
			throw new NameLengthIsNotWithinTheAllowedLimit(name);
		Parent = parent!;
		Name = name;
#if DEBUG
		this.callerFilePath = callerFilePath;
		this.callerLineNumber = callerLineNumber;
		this.callerMemberName = callerMemberName;
#endif
		FullName = string.IsNullOrEmpty(parent?.Name)
			? name
			: parent.FullName + ParentSeparator + name;
	}

#if DEBUG
	protected readonly string callerFilePath;
	protected readonly int callerLineNumber;
	protected readonly string callerMemberName;
#endif
	public const char ParentSeparator = '/';

	private static bool IsNotMethodOrPackageAndNotConflictingType(Context context, Context parent,
		string name)
	{
		var lastLetterNumber = -1;
		return context is not Method && context is not Package &&
			!name.IsWordOrWordWithNumberAtEnd(out lastLetterNumber) ||
			lastLetterNumber != -1 && parent.FindTypeCore(name[..^1]) != null;
	}

	public sealed class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name)
		: Exception(name);

	public sealed class PackageNameMustBeAWordWithoutSpecialCharacters(string name) : Exception(
		"Name " + name + "; Must start with a letter, then only allowed characters are: Letters " +
		"(A-z), Numbers (0-9) or '-'. Do not use '.', '_', spaces or any other special characters.");

	public Context Parent { get; }
	public string Name { get; }
	public string FullName { get; }

	public abstract Type? FindTypeCore(string name, Context? searchingFrom = null);
	public Type GetType(string name) => FindType(name) ?? throw new TypeNotFound(name, this);

	public Type? FindType(string name)
	{
		lock (types)
		{
			FindTypeCount++;
			if (types.TryGetValue(name, out var type))
				return type;
			var result = GuessTypeFromName();
			if (result != null)
				types[name] = result;
			return result;
		}

		Type? GuessTypeFromName()
		{
			if (name == Name && this is Type || this is Type && ((Type)this).IsGeneric &&
				name.StartsWith(Name, StringComparison.Ordinal) &&
				name == Name + GenericImplementationPostfix)
				return (Type)this;
			if (name.StartsWith("List", StringComparison.Ordinal) && name.Length > 4 && name[4] != '(')
				throw new ListPrefixIsNotAllowedUseImplementationTypeNameInPlural(name);
			if (name.EndsWith('s'))
				return TryGetTypeFromPluralNameAsListWithSingularName(name);
			if (name.EndsWith(')') && name.Contains('('))
				return TryGetGenericTypeWithArguments(name);
			return FindFullType(name) ?? FindTypeCore(name, this);
		}
	}

	private readonly IDictionary<string, Type?> types = new Dictionary<string, Type?>();
	internal static int FindTypeCount;

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
		if (singularName == Type.GenericUppercase)
			return GetType(Type.List);
		var elementType = FindFullType(singularName);
		if (elementType == null && singularName.Length > 0 && char.IsUpper(singularName[0]))
			elementType = FindTypeCore(singularName, this);
		if (elementType != null)
			return GetListImplementationType(elementType);
		return FindFullType(name) ?? FindTypeCore(name, this);
	}

	private const string GenericImplementationPostfix = "(" + Type.GenericUppercase + ")";

	private Type? TryGetGenericTypeWithArguments(string name)
	{
		var mainTypeName = name[..name.IndexOf('(')];
		if (mainTypeName.Length == 0)
			return null;
		var mainType = FindTypeCore(mainTypeName, this);
		if (mainType == null)
			return null;
		var rest = name[(mainType.Name.Length + 1)..^1];
		var arguments = rest.Split(',', StringSplitOptions.TrimEntries);
		if (rest.Contains(Type.GenericUppercase))
		{
			var namedTypes = GetNamedTypes(mainType, arguments);
			return mainType.Package.FindDirectType(mainType.GetImplementationName(namedTypes)) ??
				new GenericType(mainType, namedTypes);
		}
		var argumentTypes = TryGetArgumentTypes(arguments);
		return argumentTypes == null ? null : mainType.GetGenericImplementation(argumentTypes);
	}

	private static NamedType[] GetNamedTypes(Type mainType, IReadOnlyList<string> argumentTypeNames)
	{
		var namedTypes = new NamedType[argumentTypeNames.Count];
		for (var index = 0; index < argumentTypeNames.Count; index++)
			namedTypes[index] = new Parameter(mainType, argumentTypeNames[index]);
		return namedTypes;
	}

	private Type[]? TryGetArgumentTypes(IReadOnlyList<string> argumentTypeNames)
	{
		var argumentTypes = new Type[argumentTypeNames.Count];
		for (var index = 0; index < argumentTypeNames.Count; index++)
		{
			var found = FindType(argumentTypeNames[index]);
			if (found == null)
				return null;
			argumentTypes[index] = found;
		}
		return argumentTypes;
	}

	public sealed class TypeArgumentsCountDoesNotMatchGenericType(Type mainType,
		IReadOnlyList<Type> typeArguments) : Exception("The generic type " + mainType +
		" needs these type arguments: " + mainType.GetGenericTypeArguments().ToList().ToBrackets() +
		", this does not match provided types: " + typeArguments.ToBrackets());

	public GenericTypeImplementation GetListImplementationType(Type implementation) =>
		GetType(Type.List).GetGenericImplementation(implementation);

	public GenericTypeImplementation GetDictionaryImplementationType(Type keyType, Type valueType) =>
		GetType(Type.Dictionary).GetGenericImplementation(keyType, valueType);

	private Type? FindFullType(string name) =>
		name.Contains(ParentSeparator)
			? name == FullName
				? this as Type
				: GetPackage()?.FindFullType(name)
			: null;

	public override string ToString() => FullName;

	//ncrunch: no coverage start
	public string ToDebugString() =>
#if DEBUG
		FullName + " Parent+" + Parent +
		", created from " + callerMemberName + " in " + callerFilePath + ":line " + callerLineNumber;
#else
		FullName + " Parent+" + Parent;
#endif
	//ncrunch: no coverage end

	public Package? GetPackage() =>
		this is Package package
			// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
			? Parent is null
				? null
				: package
			: Parent.GetPackage();

	public sealed class TypeNotFound(string typeName, Context context)
		: Exception($"{typeName} not found in\n" + WriteContextTypes(context))
	{
		private static string WriteContextTypes(Context context)
		{
			var result = context.GetType().Name + " " + context.FullName + ", " +
				"available types: " + string.Join(", ", context.types.Keys);
			// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
			if (context.Parent != null && context.Parent.Name != string.Empty)
				result += "\n\tParent " + WriteContextTypes(context.Parent);
			return result;
		}
	}
}