using System;
using System.Runtime.CompilerServices;

namespace Strict.Language;

/// <summary>
/// Keeps all known types for use, if in <see cref="Package"/> contains all known types
/// and traits the context is inside a type, all members are available as
/// well, in a method more scope information is available. The high level context knows it all.
/// </summary>
public abstract class Context
{
	protected Context(Context? parent, string name)
	{
		if (parent != null && (string.IsNullOrWhiteSpace(name) ||
			this is not Method && !name.IsWord()))
			throw new NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(name);
		Parent = parent!;
		Name = name;
	}

	public sealed class NameMustBeAWordWithoutAnySpecialCharactersOrNumbers : Exception
	{
		public NameMustBeAWordWithoutAnySpecialCharactersOrNumbers(string name) : base(name) { }
	}

	public Context Parent { get; }
	public string Name { get; }

	/// <summary>
	/// Could be optimized in the future for contexts that are used a lot (10+ calls) and have at
	/// least 5+ types in its package. A dictionary could cache the same name calls (e.g. Number,
	/// Text, etc. or even Base.Log always return the same type).
	/// </summary>
	public Type GetType(string name)
	{
		// Generics still need to be supported (see Log.strict for Output<text>)
		// https://deltaengine.fogbugz.com/f/cases/24562
		if (name.StartsWith("Iterator<", StringComparison.Ordinal))
			name = name.Split('<', '>')[1];
		if (name.Contains('<'))
			name = name.Split('<')[0];
		// Arrays are also not supported yet, simply return base type, however only if we do not find a name ending with s already and do proper array fun
		// https://deltaengine.fogbugz.com/f/cases/24563/
		if (name.EndsWith('s'))
			name = name[..^1];
		if (name == Name)
			return (Type)this;
		return (FindFullType(name) ?? FindType(name, this)) ??
			throw new TypeNotFound(name, ToString());
	}

	private Type? FindFullType(string name) =>
		name.Contains('.')
			? name == ToString()
				? this as Type
				: GetPackage()?.FindFullType(name)
			: null;

	public override string ToString() =>
		// ReSharper disable once ConditionalAccessQualifierIsNonNullableAccordingToAPIContract
		(string.IsNullOrEmpty(Parent?.Name)
			? ""
			: Parent + ".") + Name;

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