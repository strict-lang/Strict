using System;

namespace Strict.Language;

public sealed class Parameter : NamedType
{
	public Parameter(Type parentType, string name, Expression defaultValue) : base(parentType, name.Replace(Type.Mutable, ""),
		defaultValue.ReturnType)
	{
		DefaultValue = defaultValue;
		IsMutable = name.Contains(Type.Mutable, StringComparison.Ordinal);
	}

	public Expression? DefaultValue { get; }

	public Parameter(Type parentType, string nameAndType) : base(parentType,
		nameAndType.Replace(Type.Mutable, "")) =>
		IsMutable = nameAndType.Contains(Type.Mutable, StringComparison.Ordinal);

	public Parameter CloneWithImplementationType(Type newType)
	{
		if (Type == newType)
			return this;
		var clone = (Parameter)MemberwiseClone();
		clone.Type = newType;
		return clone;
	}
}