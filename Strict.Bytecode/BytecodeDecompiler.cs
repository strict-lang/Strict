using System.IO.Compression;
using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Language;

namespace Strict.Bytecode;

/// <summary>
/// Partially reconstructs .strict source files from a .strict_binary ZIP file.
/// The output is a best-effort approximation — it may not compile, contains no tests,
/// and only includes what the serialized instructions reveal. Useful for diagnostics/debugging.
/// </summary>
public static class BytecodeDecompiler
{
	/// <summary>
	/// Opens a .strict_binary ZIP file, deserializes each bytecode entry, and writes
	/// a reconstructed .strict source file per entry into <paramref name="outputFolder" />.
	/// </summary>
	public static void Decompile(string strictBinaryFilePath, string outputFolder, Package package)
	{
		Directory.CreateDirectory(outputFolder);
		using var zip = ZipFile.OpenRead(strictBinaryFilePath);
		foreach (var entry in zip.Entries)
		{
			if (!entry.Name.EndsWith(".bytecode", StringComparison.OrdinalIgnoreCase))
				continue;
			var typeName = Path.GetFileNameWithoutExtension(entry.Name);
			List<Instruction> instructions;
			try
			{
				using var entryStream = entry.Open();
				instructions = BytecodeSerializer.DeserializeEntry(entryStream, package);
			}
			catch (Exception ex)
			{
				WriteErrorFile(outputFolder, typeName, ex.Message);
				continue;
			}
			var sourceLines = ReconstructSource(typeName, instructions);
			var outputPath = Path.Combine(outputFolder, typeName + ".strict");
			File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
		}
	}

	private static void WriteErrorFile(string outputFolder, string typeName, string errorMessage)
	{
		var lines = new[] { "// Decompilation failed: " + errorMessage };
		File.WriteAllLines(Path.Combine(outputFolder, typeName + ".strict"), lines, Encoding.UTF8);
	}

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
					storeVar.ValueInstance.GetTypeExceptText().Name);
				break;
			case StoreVariableInstruction storeVar:
				bodyLines.Add("\tconstant " + storeVar.Identifier + " = " +
					storeVar.ValueInstance.ToExpressionCodeString());
				break;
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
