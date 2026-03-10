using System.IO.Compression;
using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Bytecode;

/// <summary>
/// Partially reconstructs .strict source files from a .strictbinary ZIP file as an approximation.
/// For debugging, will not compile, no tests. Only includes what serialized data reveals.
/// </summary>
public sealed class BytecodeDecompiler(Package basePackage)
{
	/// <summary>
	/// Opens a .strictbinary ZIP file, deserializes each bytecode entry, and writes
	/// a reconstructed .strict source file per entry into <paramref name="outputFolder" />.
	/// </summary>
	public void Decompile(string strictBinaryFilePath, string outputFolder)
	{
		Directory.CreateDirectory(outputFolder);
		using var zip = ZipFile.OpenRead(strictBinaryFilePath);
		foreach (var entry in zip.Entries)
			if (entry.Name.EndsWith(".bytecode", StringComparison.OrdinalIgnoreCase))
			{
				var typeName = Path.GetFileNameWithoutExtension(entry.Name);
				List<Instruction> instructions;
				try
				{
					using var entryStream = entry.Open();
					instructions = BytecodeSerializer.DeserializeEntry(entryStream, basePackage);
				}
				//ncrunch: no coverage start
				catch (Exception ex)
				{
					WriteErrorFile(outputFolder, typeName, ex.Message);
					continue;
				} //ncrunch: no coverage end
				var sourceLines = ReconstructSource(typeName, instructions);
				var outputPath = Path.Combine(outputFolder, typeName + ".strict");
				File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
			}
	}

	//ncrunch: no coverage start
	private static void WriteErrorFile(string outputFolder, string typeName, string errorMessage)
	{
		var lines = new[] { "// Decompilation failed: " + errorMessage };
		File.WriteAllLines(Path.Combine(outputFolder, typeName + ".strict"), lines, Encoding.UTF8);
	} //ncrunch: no coverage end

	private static IEnumerable<string> ReconstructSource(string typeName,
		IList<Instruction> instructions)
	{
		var lines = new List<string> { "// Decompiled from " + typeName + ".bytecode" };
		var members = new List<string>();
		var bodyLines = new List<string>();
		foreach (var inst in instructions)
		{
			switch (inst)
			{
			case StoreVariableInstruction storeVar when storeVar.IsMember:
				members.Add("has " + storeVar.Identifier + " " +
					storeVar.ValueInstance.GetType().Name);
				break;
			case StoreVariableInstruction storeVar:
				//ncrunch: no coverage start
				bodyLines.Add("\tconstant " + storeVar.Identifier + " = " +
					storeVar.ValueInstance.ToExpressionCodeString());
				break; //ncrunch: no coverage end
			case LoadVariableToRegister loadVar:
				bodyLines.Add("\t// load " + loadVar.Identifier + " → " + loadVar.Register);
				break;
			case BinaryInstruction bin when bin.Registers.Length >= 2:
				bodyLines.Add("\t// " + bin.InstructionType + " " +
					string.Join(", ", bin.Registers.Select(r => r.ToString())));
				break;
			case ReturnInstruction ret:
				bodyLines.Add("\t// return " + ret.Register);
				break;
			//ncrunch: no coverage start
			case Invoke invoke:
				bodyLines.Add("\t// invoke " + (invoke.Method?.Method.Name ?? "?") + " → " +
					invoke.Register);
				break;
			case LoopBeginInstruction loopBegin:
				bodyLines.Add("\t// for loop start " + loopBegin.Register +
					(loopBegin.IsRange
						? " to " + loopBegin.EndIndex
						: ""));
				break;
			case LoopEndInstruction:
				bodyLines.Add("\t// for loop end");
				break;
			default:
				bodyLines.Add("\t// " + inst);
				break;
			 //ncrunch: no coverage end
			}
		}
		lines.AddRange(members);
		if (bodyLines.Count > 0)
		{
			lines.Add("Run");
			lines.AddRange(bodyLines);
		}
		return lines;
	}
}
