using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;

namespace Strict.Bytecode;

/// <summary>
/// Partially reconstructs .strict source files from StrictBinary (e.g. from .strictbinary) as an
/// approximation. For debugging, will not compile, no tests. Only includes what bytecode reveals.
/// </summary>
public sealed class BytecodeDecompiler
{
	/// <summary>
	/// Opens a .strictbinary ZIP file, deserializes each bytecode entry, and writes
	/// a reconstructed .strict source file per entry into <paramref name="outputFolder" />.
	/// </summary>
	public void Decompile(StrictBinary allInstructions, string outputFolder)
	{
		Directory.CreateDirectory(outputFolder);
		foreach (var typeMethods in allInstructions.MethodsPerType)
		{
			var sourceLines = ReconstructSource(typeMethods.Value);
			var outputPath = Path.Combine(outputFolder, typeMethods.Key + ".strict");
			File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
		}
	}

	private static IReadOnlyList<string> ReconstructSource(BytecodeMembersAndMethods typeData)
	{
		var lines = new List<string>();
		foreach (var member in typeData.Members)
			lines.Add("has " + member.Name + " " + member.FullTypeName +
				(member.InitialValueExpression != null
					? " = " + member.InitialValueExpression
					: ""));
		foreach (var (methodName, methods) in typeData.InstructionsPerMethodGroup)
		foreach (var method in methods)
		{
			lines.Add(BytecodeMembersAndMethods.ReconstructMethodName(methodName, method));
			var bodyLines = new List<string>();
			for (var index = 0; index < method.Instructions.Count; index++)
			{
				switch (method.Instructions[index])
				{
				case StoreVariableInstruction storeVar:
					bodyLines.Add("\tconstant " + storeVar.Identifier + " = " +
						storeVar.ValueInstance.ToExpressionCodeString());
					break;
				case Invoke invoke when invoke.Method != null &&
					index + 1 < method.Instructions.Count &&
					method.Instructions[index + 1] is StoreFromRegisterInstruction nextStore &&
					nextStore.Register == invoke.Register:
					bodyLines.Add("\tconstant " + nextStore.Identifier + " = " + invoke.Method);
					index++;
					break;
				case Invoke { Method: not null } invoke:
					bodyLines.Add("\t" + invoke.Method);
					break;
				//TODO: most instructions are still missing here!
				}
			}
			lines.AddRange(bodyLines);
		}
		return lines;
	}
}