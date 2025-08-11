#if TODO
using Strict.Expressions;
using Strict.Language;

namespace Strict.Validators;

/// <summary>
/// Visitor that finds all mutable assignments for a specific type/variable
/// </summary>
public sealed class MutableAssignmentVisitor(NamedType typeToSearchFor) : Visitor
{
	private readonly List<MutableReassignment> foundAssignments = new();
	
	protected override void HandleExpression(Expression expression)
	{
		if (expression is MutableReassignment reassignment && 
		    reassignment.Name == typeToSearchFor.Name && 
		    reassignment.ReturnType == typeToSearchFor.Type)
		{
			foundAssignments.Add(reassignment);
		}
	}
	
	/// <summary>
	/// Get all found mutable assignments
	/// </summary>
	public IReadOnlyList<MutableReassignment> GetFoundAssignments() => foundAssignments.AsReadOnly();
	
	/// <summary>
	/// Check if any mutable assignments were found
	/// </summary>
	public bool HasAnyAssignments() => foundAssignments.Count > 0;
	
	/// <summary>
	/// Legacy method for backward compatibility
	/// </summary>
	[Obsolete("Use HasAnyAssignments() instead")]
	public override Expression? Visit(Expression searchIn)
	{
		VisitExpression(searchIn);
		return foundAssignments.FirstOrDefault();
	}
}
#endif