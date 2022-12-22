using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Xml.Linq;

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
		if (parent != null && this is not GenericType && (string.IsNullOrWhiteSpace(name) ||
			this is not Method && !name.IsWord()))
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		Parent = parent!;
		Name = name;
		FullName = string.IsNullOrEmpty(parent?.Name) || parent.Name is nameof(Base)
			? name
			: parent + "." + name;
	}

	public sealed class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers : Exception
	{
		public NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name) : base(name) { }
	}

	public Context Parent { get; }
	public string Name { get; }
	public string FullName { get; }

	/// <summary>
	/// TODO: Could be optimized in the future for contexts that are used a lot (10+ calls) and have at
	/// least 5+ types in its package. A dictionary could cache the same name calls (e.g. Number,
	/// Text, etc. or even Base.Log always return the same type).
	/// TODO: check type limit stuff, will make parsing much more complex!
	///characters(2) ???? not a type
	///has Characters(2) -> limit???
	/// </summary>
	public Type GetType(string name)
	{
		if (name == Name)
			return (Type)this;
		if (name.Contains('(') && name.EndsWith(')'))
			return GetGenericTypeWithArguments(name);
		if (!name.EndsWith('s'))
			return (FindFullType(name) ?? FindType(name, this)) ??
				throw new TypeNotFound(name, FullName);
		var singularName = name[..^1];
		if (singularName == Base.Generic)
			// ReSharper disable once TailRecursiveCall
			return GetType(Base.List);
		var elementType = FindFullType(singularName) ?? FindType(singularName, this);
		if (elementType != null)
			return GetListImplementationType(elementType);
		return (FindFullType(name) ?? FindType(name, this)) ?? throw new TypeNotFound(name, FullName);
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

	public sealed class TypeArgumentsDoNotMatchGenericType : Exception
	{
		public TypeArgumentsDoNotMatchGenericType(Type mainType,
			IReadOnlyCollection<Type> typeArguments) : base($"The generic type {
				mainType.Name
			} needs these type arguments: {
				mainType.Members.Where(m => m.Type.IsGeneric).ToList().ToBrackets()
			}, does not match provided types: {
				typeArguments.ToBrackets()
			}") { }
	}

	public GenericType GetListImplementationType(Type implementation) =>
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