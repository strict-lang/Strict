using System.IO.Compression;
using System.Runtime.CompilerServices;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

[assembly: InternalsVisibleTo("Strict")]
[assembly: InternalsVisibleTo("Strict.Optimizers")]

namespace Strict.Bytecode;

public sealed partial class BinaryExecutable
{
	/// <summary>
	/// Before reading instructions, create embedded type stubs so generic constants
	/// can resolve their element types without loading any source.
	/// </summary>
	private void PopulateStubTypesFromEmbeddedEntries(IEnumerable<string> typeNames)
	{
		var names = typeNames.ToList();
		foreach (var typeName in names.Where(name => !name.Contains(Context.ParentSeparator)))
			EnsureStubType(TypeResolver, typeName);
		foreach (var typeFullName in names.Where(name => name.Contains(Context.ParentSeparator)))
			EnsureStubType(basePackage, GetSimpleTypeName(typeFullName));
		noneType = basePackage.GetType(Type.None);
		booleanType = basePackage.GetType(Type.Boolean);
		numberType = basePackage.GetType(Type.Number);
		if (basePackage.FindDirectType(Type.Character) != null)
			characterType = basePackage.GetType(Type.Character);
		if (basePackage.FindDirectType(Type.Range) != null)
			rangeType = basePackage.GetType(Type.Range);
		if (basePackage.FindDirectType(Type.List) != null)
			listType = basePackage.GetType(Type.List);
	}

	private void EnsureStubType(Package package, string name)
	{
		var genericStart = name.IndexOf('(');
		var mainName = genericStart < 0
			? name
			: name[..genericStart];
		if (genericStart < 0)
		{
			if (package.FindDirectType(mainName) == null)
				new Type(package, new TypeLines(mainName));
			return;
		}
		var arguments = SplitGenericArguments(name[(genericStart + 1)..^1]);
		foreach (var argument in arguments)
			if (TypeResolver.FindType(argument) == null)
				EnsureStubType(basePackage, argument);
		if (package.FindDirectType(mainName) != null)
			return;
		EnsureStubType(basePackage, Type.GenericUppercase);
		new Type(package, new TypeLines(mainName, arguments.Select((_, index) =>
				Type.HasWithSpaceAtEnd + "generic" + (char)('A' + index) + " " + Type.GenericUppercase).
			ToArray())).ParseMembersAndMethods(new MethodExpressionParser());
	}

	private static List<string> SplitGenericArguments(string arguments)
	{
		var result = new List<string>();
		var depth = 0;
		var argumentStart = 0;
		for (var index = 0; index < arguments.Length; index++)
			if (arguments[index] == '(')
				depth++;
			else if (arguments[index] == ')')
				depth--;
			else if (arguments[index] == ',' && depth == 0)
			{
				result.Add(arguments[argumentStart..index].Trim());
				argumentStart = index + 1;
			}
		result.Add(arguments[argumentStart..].Trim());
		return result;
	}

	private void RestoreEmbeddedMembers()
	{
		foreach (var (typeName, binaryType) in MethodsPerType)
		{
			var type = ResolveType(typeName);
			if (type.Members.Count > 0)
				continue;
			foreach (var member in binaryType.Members)
			{
				var memberType = ResolveType(member.FullTypeName);
				type.Members.Add(new Member(type, member.Name, memberType,
					usedKeyword: member.IsConstant
						? Keyword.Constant
						: Keyword.Has)
				{
					InitialValue = member.InitialValueExpression is SetInstruction constant
						? new Value(memberType, constant.ValueInstance)
						: null
				});
			}
			RestoreTraitMethods(type, binaryType);
		}
	}

	private void RestoreTraitMethods(Type type, BinaryType binaryType)
	{
		if (type.Methods.Count > 0 || !type.IsTrait || !IsSignatureOnly(binaryType))
			return;
		var parser = new MethodExpressionParser();
		var lineNumber = 1;
		foreach (var overloads in binaryType.MethodGroups.Values)
			foreach (var method in overloads)
			{
				if (!CanResolveTraitMethod(method))
					continue;
				type.Methods.Add(new Method(type, lineNumber++, parser, [BuildTraitMethodHeader(method)]));
			}
	}

	private static bool IsSignatureOnly(BinaryType binaryType)
	{
		var sawMethod = false;
		foreach (var overloads in binaryType.MethodGroups.Values)
			foreach (var method in overloads)
			{
				sawMethod = true;
				if (method.instructions.Count > 0)
					return false;
			}
		return sawMethod;
	}

	private bool CanResolveTraitMethod(BinaryMethod method)
	{
		if (method.Name != Method.From && !CanResolveStoredType(method.ReturnTypeName))
			return false;
		foreach (var parameter in method.parameters)
			if (!CanResolveStoredType(parameter.FullTypeName))
				return false;
		return true;
	}

	private bool CanResolveStoredType(string typeName)
	{
		var simple = GetSimpleTypeName(typeName);
		if (simple.Length == 0 || simple == Type.None)
			return true;
		if (simple.Contains("Generic", StringComparison.Ordinal))
			return false;
		try
		{
			EnsureTypePieces(simple);
			ResolveType(simple);
			return true;
		}
		catch (Context.TypeNotFound)
		{
			return false;
		}
	}

	private void EnsureTypePieces(string simple)
	{
		var open = simple.IndexOf('(');
		if (open > 0 && simple.EndsWith(')'))
		{
			foreach (var part in simple[(open + 1)..^1].Split(',',
				StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
				EnsureTypePieces(GetSimpleTypeName(part));
			return;
		}
		if (TypeResolver.FindType(simple) != null)
			return;
		if (simple.EndsWith('s') && simple.Length > 1)
		{
			var singular = simple[..^1];
			if (char.IsUpper(singular[0]) && basePackage.FindDirectType(singular) == null)
				new Type(basePackage, new TypeLines(singular));
			return;
		}
		if (basePackage.FindDirectType(simple) == null)
			new Type(basePackage, new TypeLines(simple));
	}

	private static string BuildTraitMethodHeader(BinaryMethod method)
	{
		var header = method.Name;
		if (method.parameters.Count > 0)
			header += "(" + string.Join(", ", method.parameters.Select(parameter =>
				parameter.Name + " " + GetSimpleTypeName(parameter.FullTypeName))) + ")";
		var returnTypeName = GetSimpleTypeName(method.ReturnTypeName);
		if (method.Name != Method.From && returnTypeName.Length > 0 && returnTypeName != Type.None)
			header += " " + returnTypeName;
		return header;
	}

	//TODO: avoid! remove!
	internal Type ResolveType(string typeName)
	{
		var isPrefixed = typeName.Contains(Context.ParentSeparator);
		var package = isPrefixed
			? basePackage
			: TypeResolver;
		var resolved = TypeResolver.FindType(typeName) ?? (isPrefixed
			? package.FindFullType(typeName) ?? package.FindType(GetSimpleTypeName(typeName))
			: null);
		if (resolved != null)
			return resolved;
		if (typeName.EndsWith(')') && typeName.Contains('('))
		{
			var genericName = GetGenericLookupName(typeName);
			if (entryPackage != null)
				EnsureStubType(package, genericName);
			return TypeResolver.GetType(genericName);
		}
		if (char.IsLower(typeName[0]))
			throw new TypeNotFoundForBytecode(typeName);
		var simpleTypeName = GetSimpleTypeName(typeName);
		EnsureTypeExists(package, simpleTypeName);
		return package.GetType(simpleTypeName);
	}

	private static string GetSimpleTypeName(string typeName) =>
		typeName.Contains(Context.ParentSeparator)
			? typeName[(typeName.LastIndexOf(Context.ParentSeparator) + 1)..]
			: typeName;

	private static string GetGenericLookupName(string typeName)
	{
		var openParenIndex = typeName.IndexOf('(');
		var mainTypeName = typeName[..openParenIndex];
		var simpleMainTypeName = mainTypeName.Contains(Context.ParentSeparator)
			? mainTypeName[(mainTypeName.LastIndexOf(Context.ParentSeparator) + 1)..]
			: mainTypeName;
		return simpleMainTypeName + typeName[openParenIndex..];
	}

	public sealed class TypeNotFoundForBytecode(string typeName)
		: Exception("Type '" + typeName + "' not found while deserializing bytecode");

	private static void EnsureTypeExists(Package package, string typeName)
	{
		if (package.FindDirectType(typeName) == null)
			new Type(package, new TypeLines(typeName, Method.Run)).ParseMembersAndMethods(
				new MethodExpressionParser());
	}
}
