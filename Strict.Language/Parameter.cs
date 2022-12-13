using System;

namespace Strict.Language;

public sealed class Parameter : NamedType
{
	public Parameter(Type parentType, string nameAndType) : base(parentType, nameAndType)
	{
		IsMutable = nameAndType.Contains("Mutable(", StringComparison.Ordinal);
	}

	public Parameter CloneWithImplementationType(Type newType)
	{
		if (Type == newType)
			return this;
		var clone = (Parameter)MemberwiseClone();
		clone.Type = newType;
		return clone;
	}
}