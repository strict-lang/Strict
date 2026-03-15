using System.IO.Compression;
using Strict.Bytecode.Instructions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Serialization;

/// <summary>
/// After <see cref="BytecodeGenerator"/> generates all bytecode from the parsed expressions or
/// <see cref="BytecodeDeserializer"/> loads a .strictbinary ZIP file with the same bytecode,
/// this class contains the deserialized bytecode for each type used with each method used.
/// </summary>
public sealed class StrictBinary
{
	/// <summary>
	/// Each key is a type.FullName (e.g. Strict/Number, Strict/ImageProcessing/Color), the Value
	/// contains all members of this type and all not stripped out methods that were actually used.
	/// </summary>
	public Dictionary<string, BytecodeMembersAndMethods> MethodsPerType = new();

	/// <summary>
	/// Writes optimized <see cref="Instruction" /> lists per type into a compact .strictbinary ZIP.
	/// The ZIP contains one entry per type named {typeName}.bytecode.
	/// Entry layout: magic(6) + version(1) + string-table + instruction-count(7bit) + instructions.
	/// </summary>
	public void Serialize(string filePath)
	{
		using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
		using var zip = new ZipArchive(fileStream, ZipArchiveMode.Create, leaveOpen: false);
		foreach (var (fullTypeName, membersAndMethods) in MethodsPerType)
		{
			var entry = zip.CreateEntry(fullTypeName + BytecodeEntryExtension, CompressionLevel.Optimal);
			using var entryStream = entry.Open();
			using var writer = new BinaryWriter(entryStream);
			membersAndMethods.Write(writer);
		}
	}

	public const string Extension = ".strictbinary";
	public const string BytecodeEntryExtension = ".bytecode";

	public IReadOnlyList<Instruction>? FindInstructions(Type type, Method method) =>
		FindInstructions(type.FullName, method.Name, method.Parameters.Count, method.ReturnType.Name);

	public IReadOnlyList<Instruction>? FindInstructions(string fullTypeName, string methodName,
		int parametersCount, string returnType = "") =>
		MethodsPerType.TryGetValue(fullTypeName, out var methods)
			? methods.InstructionsPerMethodGroup.GetValueOrDefault(methodName)?.Find(m =>
				m.Parameters.Count == parametersCount && m.ReturnTypeName == returnType)?.Instructions
			: null;

	//public static string GetMethodKey(string name, int parametersCount, string returnType) =>
	//	name + parametersCount + returnType;

	//public static string GetMethodKey(Method method) =>
	//	GetMethodKey(method.Name, method.Parameters.Count, method.ReturnType.Name);


}