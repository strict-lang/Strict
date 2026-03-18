using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Bytecode.Serialization;

//TODO: remove again, this is plain stupid!
/// <summary>
/// Compatibility wrapper around <see cref="BinaryExecutable"/> for loading bytecode from files.
/// </summary>
public sealed class BytecodeDeserializer(string filePath)
{
	public BytecodeDeserializerResult Deserialize(Package basePackage) =>
		new(new BinaryExecutable(filePath, basePackage));
}

public sealed class BytecodeDeserializerResult(BinaryExecutable binary)
{
	public List<Instruction>? Find(string typeName, string methodName, int parameterCount)
	{
		foreach (var (fullName, typeData) in binary.MethodsPerType)
		{
			var shortName = fullName.Contains('/') ? fullName[(fullName.LastIndexOf('/') + 1)..] : fullName;
			if (!string.Equals(shortName, typeName, StringComparison.Ordinal) &&
				!string.Equals(fullName, typeName, StringComparison.Ordinal))
				continue;
			var methods = typeData.MethodGroups.GetValueOrDefault(methodName);
			if (methods == null)
				continue;
			var method = methods.Find(m => m.parameters.Count == parameterCount);
			if (method != null)
				return method.instructions;
		}
		return null;
	}
}
