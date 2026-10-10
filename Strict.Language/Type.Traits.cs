namespace Strict.Language;

public partial class Type
{
	public sealed class MustImplementAllTraitMethodsOrNone(Type type,
		string traitName,
		IEnumerable<Method> missingTraitMethods) : ParsingFailed(type, type.typeParser.LineNumber,
		"Trait Type:" + traitName + " Missing methods: " + string.Join(", ", missingTraitMethods));

	private void CheckIfTraitIsImplementedFullyOrNone(Type trait)
	{
		var traitMethods = GetRequiredTraitMethods(trait).
			Where(traitMethod => traitMethod.Name != Method.From).ToList();
		var nonImplementedTraitMethods = traitMethods.Where(traitMethod =>
			traitMethod.Name != Method.From &&
			methods.All(implementedMethod => traitMethod.Name != implementedMethod.Name)).ToList();
		if (nonImplementedTraitMethods.Count > 0 &&
			nonImplementedTraitMethods.Count != traitMethods.Count)
			throw new MustImplementAllTraitMethodsOrNone(this, trait.Name, nonImplementedTraitMethods);
	}

	private static IEnumerable<Method> GetRequiredTraitMethods(Type trait)
	{
		foreach (var method in trait.Methods)
			yield return method;
		foreach (var member in trait.Members)
			if (IsTraitRequirementMember(member))
				foreach (var method in GetRequiredTraitMethods(member.Type))
					yield return method;
	}

	public bool IsTrait =>
		!IsNumber && !IsBoolean && !IsText && CheckIfParsed() && CanBeTraitBasedOnMembers &&
		Methods.All(IsTraitMethodDeclaration);

	internal bool CanBeTraitBasedOnMembers =>
		!IsNumber && !IsBoolean && (Members.Count == 0 || Members.All(IsTraitRequirementMember) &&
			(Members.Any(member => !member.IsPublic) || Members.Count > 1));
	internal bool MustUseBodylessTraitMethods =>
		!IsNumber && !IsBoolean && (Members.Count == 0 ||
			Members.Count > 1 && Members.All(IsTraitCompositionMember));

	internal static bool IsTraitMethodDeclaration(Method method) => method.lines.Count == 1;

	private static bool IsTraitRequirementMember(Member member) =>
		member.Type != member.DefinedIn && member.Type.IsTrait;

	private static bool IsTraitCompositionMember(Member member) =>
		member.IsPublic && member.Name == member.Type.Name && member.Type != member.DefinedIn &&
		member.Type.IsTrait;

	public sealed class TypeHasNoMembersAndThusMustBeATraitWithoutMethodBodies(Type type)
		: ParsingFailed(type, 0);
}
