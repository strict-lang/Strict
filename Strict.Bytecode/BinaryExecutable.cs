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

/// <summary>
/// Loads <see cref="Instruction" /> bytecode for each type used with each method used. Generated
/// from <see cref="BinaryGenerator"/> or loaded from a compact .strictbinary ZIP file, which is
/// done via <see cref="Serialize(string)"/>. Used by the VirtualMachine or executable generation.
/// </summary>
public sealed partial class BinaryExecutable(Package basePackage)
{
	internal readonly Package basePackage = basePackage;

	internal Type noneType = basePackage.FindType(Type.None) ??
		new Type(basePackage, new TypeLines(Type.None));

	internal Type booleanType = basePackage.FindType(Type.Boolean) ??
		new Type(basePackage, new TypeLines(Type.Boolean));

	internal Type numberType = basePackage.FindType(Type.Number) ??
		new Type(basePackage, new TypeLines(Type.Number));

	internal Type characterType = basePackage.FindType(Type.Character) ??
		new Type(basePackage, new TypeLines(Type.Character));

	internal Type rangeType = basePackage.FindType(Type.Range) ??
		new Type(basePackage, new TypeLines(Type.Range));

	internal Type listType = basePackage.FindType(Type.List) ?? new Type(basePackage,
		new TypeLines(Type.List, Type.HasWithSpaceAtEnd + Type.GenericUppercase));

	/// <summary>
	/// Loads a fully self-contained .strictbinary without needing any external package.
	/// Builds an internal package of stub types from the embedded type entries so the VM
	/// can resolve type identity checks (IsNumber, IsList, etc.) without any source files.
	/// </summary>
	public BinaryExecutable(string filePath) : this(filePath, new Package(null, nameof(Strict))) { }

	/// <summary>
	/// Reads a .strictbinary ZIP containing all type bytecode (used types, members, methods) and
	/// instruction bodies for each type.
	/// </summary>
	public BinaryExecutable(string filePath, Package basePackage) : this(basePackage)
	{
		try
		{
			if (basePackage.FindType(Type.Any) == null)
				new Type(basePackage,
					new TypeLines(Type.Any, Method.From, BinaryOperator.To + " Type",
						BinaryOperator.To + " " + Type.Text));
			using var zip = ZipFile.OpenRead(filePath);
			if (basePackage.Parent is not Package)
			{
				entryPackage = new Package(basePackage, Path.GetDirectoryName(Path.GetFullPath(filePath))!);
				PopulateStubTypesFromEmbeddedEntries(zip.Entries.Where(entry =>
					entry.FullName.EndsWith(BinaryType.BytecodeEntryExtension,
						StringComparison.OrdinalIgnoreCase)).Select(entry => GetEntryNameWithoutExtension(entry.FullName)));
			}
			foreach (var entry in zip.Entries)
				if (entry.FullName.EndsWith(BinaryType.BytecodeEntryExtension,
					StringComparison.OrdinalIgnoreCase))
				{
					var typeFullName = GetEntryNameWithoutExtension(entry.FullName);
					using var bytecode = entry.Open();
					var reader = new BinaryReader(bytecode);
					MethodsPerType.Add(typeFullName, new BinaryType(reader, this, typeFullName));
				}
			if (basePackage.Parent is not Package)
				RestoreEmbeddedMembers();
		}
		catch (InvalidDataException ex)
		{
			throw new InvalidFile(ex.Message);
		}
	}

	/// <summary>
	/// Types of the entry package are stored without package prefix and get their own child
	/// package, so a local type like Language/Type does not merge with the base Strict/Type.
	/// </summary>
	private Package? entryPackage;

	internal Package TypeResolver => entryPackage ?? basePackage;

	private static string GetEntryNameWithoutExtension(string fullName)
	{
		var normalized = fullName.Replace('\\', '/');
		var extensionStart = normalized.LastIndexOf('.');
		return extensionStart > 0
			? normalized[..extensionStart]
			: normalized;
	}

	/// <summary>
	/// Each key is a type.FullName (e.g. Strict/Number, Strict/ImageProcessing/Color), the Value
	/// contains all members of this type and all not stripped out methods that were actually used.
	/// </summary>
	public readonly Dictionary<string, BinaryType> MethodsPerType = new();

	private BinaryMethod? entryPoint;

	public BinaryMethod EntryPoint => entryPoint ??= ResolveEntryPoint();

	public sealed class InvalidFile(string message) : Exception(message);

	public sealed class EntryPointNotFound(string what) : Exception("Entry point not found: " + what);

	public List<BinaryMethod> GetRunMethods()
	{
		var runMethods = new List<BinaryMethod>();
		foreach (var typeData in MethodsPerType.Values)
			if (typeData.MethodGroups.TryGetValue(Method.Run, out var overloads))
				runMethods.AddRange(overloads);
		return runMethods;
	}

	private BinaryMethod ResolveEntryPoint()
	{
		foreach (var typeData in MethodsPerType.Values)
			if (typeData.MethodGroups.TryGetValue(Method.Run, out var runMethods) && runMethods.Count > 0)
				return runMethods[0];
		throw new EntryPointNotFound("Run method in any type");
	}

	public List<Instruction>? FindInstructions(Type type, Method method) =>
		FindInstructions(type.FullName, method.Name, method.Parameters.Count, method.ReturnType.Name);

	public List<Instruction>? FindInstructions(string fullTypeName, string methodName,
		int parametersCount, string returnType = "") =>
		MethodsPerType.TryGetValue(fullTypeName, out var methods)
			? methods.MethodGroups.GetValueOrDefault(methodName)?.Find(method =>
				method.parameters.Count == parametersCount &&
				DoesReturnTypeMatch(method.ReturnTypeName, returnType))?.instructions
			: null;

	private static bool DoesReturnTypeMatch(string storedReturnType, string expectedReturnType) =>
		expectedReturnType.Length == 0 || storedReturnType == expectedReturnType ||
		storedReturnType.EndsWith(Context.ParentSeparator + expectedReturnType,
			StringComparison.Ordinal) ||
		expectedReturnType.EndsWith(Context.ParentSeparator + storedReturnType,
			StringComparison.Ordinal);

	public List<Instruction>? FindInstructions(string fullTypeName, string methodName,
		int parametersCount, Type returnType) =>
		FindInstructions(fullTypeName, methodName, parametersCount, returnType.Name);

	public List<Instruction> ToInstructions() => EntryPoint.instructions;

	public const string Extension = ".strictbinary";

	private readonly Dictionary<Type, Value> cachedDefaultValuesForVariableRefs = new();

	public bool UsesConsolePrint => MethodsPerType.Values.Any(type => type.UsesConsolePrint);

	public int TotalInstructionsCount =>
		MethodsPerType.Values.Sum(methods => methods.TotalInstructionCount);

	//TODO: way too complicated, fix callers.
	internal BinaryExecutable AddType(string typeFullName, List<BinaryMember> members,
		Dictionary<string, List<BinaryMethod>> methodGroups, bool isEntryType = false)
	{
		MethodsPerType[typeFullName] = new BinaryType(this, typeFullName, members, methodGroups);
		if (isEntryType && methodGroups.TryGetValue(Method.Run, out var runMethods) &&
			runMethods.Count > 0)
			entryPoint = runMethods[0];
		else if (entryPoint == null &&
			methodGroups.TryGetValue(Method.Run, out var fallbackRunMethods) &&
			fallbackRunMethods.Count > 0)
			entryPoint = fallbackRunMethods[0];
		return this;
	}

	//TODO: remove this bullshit!
	public static BinaryExecutable CreateForEntryInstructions(Package basePackage,
		List<Instruction> instructions)
	{
		var binary = new BinaryExecutable(basePackage);
		var runMethod = new BinaryMethod(Method.Run, [], Type.None, instructions);
		return binary.AddType("EntryPoint", new List<BinaryMember>(),
			new Dictionary<string, List<BinaryMethod>> { [Method.Run] = [runMethod] }, true);
	}

	internal void SetEntryPoint(string typeFullName, string methodName, int parameterCount,
		string returnTypeName)
	{
		if (!MethodsPerType.TryGetValue(typeFullName, out var typeData))
			throw new EntryPointNotFound("type " + typeFullName);
		if (!typeData.MethodGroups.TryGetValue(methodName, out var overloads))
			throw new EntryPointNotFound("method " + methodName);
		entryPoint =
			overloads.FirstOrDefault(method =>
				method.parameters.Count == parameterCount && method.ReturnTypeName == returnTypeName) ??
			throw new EntryPointNotFound("overload of " + methodName + " with " + parameterCount +
				" parameters returning " + returnTypeName);
	}

	public List<TResult> ConvertAll<TResult>(Converter<Instruction, TResult> converter) =>
		EntryPoint.instructions.Select(instruction => converter(instruction)).ToList();
}
