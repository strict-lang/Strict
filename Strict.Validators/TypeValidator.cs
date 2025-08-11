#if TODO //mostly retarded und useless here, remove most except the last method
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Validators;

/// <summary>
/// Validator that can work with packages, individual types, or collections of types
/// </summary>
public sealed class TypeValidator : Validator
{
	private readonly IEnumerable<Type>? types;
	private readonly Package? package;
	private readonly Type? singleType;

	public TypeValidator(IEnumerable<Type> types)
	{
		this.types = types;
	}

	public TypeValidator(Package package)
	{
		this.package = package;
	}

	public TypeValidator(Type type)
	{
		singleType = type;
	}

	public void Validate()
	{
		if (package != null)
		{
			var visitor = new TypeValidationVisitor();
			visitor.VisitPackage(package);
		}
		else if (singleType != null)
		{
			var visitor = new TypeValidationVisitor();
			visitor.VisitType(singleType);
		}
		else if (types != null)
		{
			foreach (var type in types)
			{
				var visitor = new TypeValidationVisitor();
				visitor.VisitType(type);
			}
		}
	}

	/// <summary>
	/// Visitor that performs type-level validation using the visitor pattern
	/// </summary>
	private sealed class TypeValidationVisitor : Visitor
	{
		public override void VisitType(Type type)
		{
			// Validate the type using sub-validators
			new MethodValidator(type.Methods).Validate();
			new MemberValidator(type).Validate();
			
			// Continue with standard traversal
			base.VisitType(type);
		}
	}
}
#endif