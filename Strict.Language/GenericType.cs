using System.Collections.Generic;

namespace Strict.Language;

/// <summary>
/// Extends a generic type with more generic details, this is NOT an implementation yet, e.g.
/// List(Generic, Generic) extends the List generic type to use two generic tuple types and force
/// implementation on that.
/// </summary>
public sealed class GenericType : Type
{
	public GenericType(Type generic, IReadOnlyList<NamedType> genericImplementations) :
		base(generic.Package,
			new TypeLines(generic.GetImplementationName(genericImplementations),
				HasWithSpaceAtEnd + generic.Name))
	{
		CreatedBy = "Generic: " + generic + ", GenericImplementations: " + genericImplementations.ToWordList() +
			", " + CreatedBy;
		Generic = generic;
		GenericImplementations = genericImplementations;
		members.AddRange(generic.Members);
		methods.AddRange(generic.Methods);
	}

	public Type Generic { get; }
	public IReadOnlyList<NamedType> GenericImplementations { get; }
}