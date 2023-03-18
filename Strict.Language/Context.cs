using System;
using System.Collections.Generic;
using System.Linq;
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
		var lastLetterNumber = -1;
		if (parent != null && this is not GenericTypeImplementation &&
			(string.IsNullOrWhiteSpace(name) ||
				this is not Method && this is not Package && !name.IsWordOrWordWithNumberAtEnd(out lastLetterNumber) ||
				HasConflictingType(parent, name, lastLetterNumber)))
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		if (this is Package && !name.IsAlphaNumericWithAllowedSpecialCharacters())
			throw new PackageNameMustBeAWordWithoutSpecialCharacters(name);
		if (!string.IsNullOrEmpty(name) && !name.Length.IsWithinLimit() &&
			!name.IsOperatorOrAllowedMethodName())
			throw new NameLengthIsNotWithinTheAllowedLimit(name);
		Parent = parent!;
		Name = name;
		FullName = string.IsNullOrEmpty(parent?.Name) || parent.Name is nameof(Base)
			? name
			: parent + "." + name;
	}

	private static bool HasConflictingType(Context context, string name, int number) =>
		number != -1 && context.FindType(name[..^1]) != null;

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
		if (name == Name)
			return (Type)this;
		if (name.StartsWith(Base.List + DoubleOpenBrackets, StringComparison.Ordinal))
			return GetNestedListType(name);
		if (name.Contains('(') && name.EndsWith(')'))
			return GetGenericTypeWithArguments(name);
		if (!name.EndsWith('s'))
			return (FindFullType(name) ?? FindType(name, this)) ??
				throw new TypeNotFound(name, FullName);
		var singularName = name[..^1];
		if (singularName == Base.Generic)
			return GetType(Base.List);
		var elementType = FindFullType(singularName) ?? FindType(singularName, this);
		if (elementType != null)
			return GetListImplementationType(elementType);
		return (FindFullType(name) ?? FindType(name, this)) ?? throw new TypeNotFound(name, FullName);
	}

	internal const string DoubleOpenBrackets = "((";

	private Type GetNestedListType(string fullName)
	{
		var list = GetType(Base.List);
		var (typeName, lines) = GetCombinedTypeNameAndLines(
			ExtractNamesWithType(fullName));
		var typeWithOtherTypesAsMembers = new Type(list.Package, new TypeLines(typeName, lines));
		return list.GetGenericImplementation(typeWithOtherTypesAsMembers);
	}

	private static string[] ExtractNamesWithType(string fullName) =>
		fullName[(Base.List.Length + DoubleOpenBrackets.Length)..^DoubleCloseBrackets.Length].
			Split(",", StringSplitOptions.TrimEntries);

	internal const string DoubleCloseBrackets = "))";

	private static (string, string[]) GetCombinedTypeNameAndLines(IReadOnlyList<string> namesWithType)
	{
		var name = "";
		var lines = new string[namesWithType.Count];
		for (var index = 0; index < namesWithType.Count; index++)
		{
			name += namesWithType[index].Split(' ')[0].MakeFirstLetterUppercase();
			lines[index] = Type.HasWithSpaceAtEnd + namesWithType[index];
		}
		return (name, lines);
	}

	private Type GetGenericTypeWithArguments(string name)
	{
		var mainType = GetType(name[..name.IndexOf('(')]);
		var argumentTypes = GetArgumentTypes(name[(mainType.Name.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries));
		return mainType.GetGenericImplementation(argumentTypes);
	}

	private List<Type> GetArgumentTypes(IEnumerable<string> argumentTypeNames)
	{
		var argumentTypes = new List<Type>();
		foreach (var argumentTypeName in argumentTypeNames)
			argumentTypes.Add(GetType(argumentTypeName));
		return argumentTypes;
	}

	public sealed class TypeArgumentsCountDoesNotMatchGenericType : Exception
	{
		public TypeArgumentsCountDoesNotMatchGenericType(Type mainType,
			IReadOnlyCollection<Type> typeArguments) : base($"The generic type {
				mainType.Name
			} needs these type arguments: {
				mainType.Members.Where(m => m.Type.IsGeneric).ToList().ToBrackets()
			}, does not match provided types: {
				typeArguments.ToBrackets()
			}") { }
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