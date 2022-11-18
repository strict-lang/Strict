using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

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
	/// Could be optimized in the future for contexts that are used a lot (10+ calls) and have at
	/// least 5+ types in its package. A dictionary could cache the same name calls (e.g. Number,
	/// Text, etc. or even Base.Log always return the same type).
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
		var listType = FindListType(name[..^1]);
		if (listType != null)
			return listType;
		return (FindFullType(name) ?? FindType(name, this)) ??
			throw new TypeNotFound(name, FullName);
	}

	private Type GetGenericTypeWithArguments(string name)
	{
		var mainType = GetType(name[..name.IndexOf('(')]);
		var argumentTypes = GetArgumentTypes(name[(mainType.Name.Length + 1)..^1].
			Split(',', StringSplitOptions.TrimEntries));
		if (mainType.Members.Count != argumentTypes.Count && mainType.FindMethodByArgumentTypes(Method.From, argumentTypes) == null)
			throw new TypeArgumentsDoNotMatchWithMainType(mainType, argumentTypes);
		return mainType.IsGeneric
			? mainType.GetGenericImplementation(argumentTypes)
			: mainType; //TODO: This needs to be constructed properly for non-generic type with Type parameters
	}

	private List<Type> GetArgumentTypes(IEnumerable<string> argumentTypeNames)
	{
		var argumentTypes = new List<Type>();
		foreach (var argumentTypeName in argumentTypeNames)
			argumentTypes.Add(GetType(argumentTypeName));
		return argumentTypes;
	}

	public sealed class TypeArgumentsDoNotMatchWithMainType : Exception
	{
		public TypeArgumentsDoNotMatchWithMainType(Type mainType,
			IReadOnlyCollection<Type> argumentTypes) : base(
			$"Argument(s) {argumentTypes.ToBrackets()} does not match type {mainType.Name}" +
			$" with constructor {mainType.Name}{mainType.Members.ToBrackets()}") { }
	}

	private Type? FindListType(string singularName)
	{
		if (singularName == Base.Generic)
			return GetType(Base.List);
		var elementType = FindFullType(singularName) ?? FindType(singularName, this);
		return elementType != null
			? GetListType(elementType)
			: null;
	}

	public GenericType GetListType(Type implementation) =>
		GetType(Base.List).GetGenericImplementation(new List<Type> { implementation });

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