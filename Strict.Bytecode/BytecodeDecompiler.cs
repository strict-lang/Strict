using System.IO.Compression;
using System.Text;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

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
		var binaryDir = Path.GetDirectoryName(Path.GetFullPath(strictBinaryFilePath)) ?? ".";
		using var zip = ZipFile.OpenRead(strictBinaryFilePath);
		foreach (var entry in zip.Entries)
			if (entry.Name.EndsWith(".bytecode", StringComparison.OrdinalIgnoreCase))
			{
				var typeName = Path.GetFileNameWithoutExtension(entry.Name);
				using var entryStream = entry.Open();
				var instructions = BytecodeSerializer.DeserializeEntry(entryStream,
					GetPackageForType(binaryDir, typeName));
				var sourceLines = ReconstructSource(instructions);
				var outputPath = Path.Combine(outputFolder, typeName + ".strict");
				File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
			}
	}

	private readonly Dictionary<string, Package> packagesByDirectory = new();

	private Package GetPackageForType(string binaryDir, string typeName)
	{
		if (basePackage.FindType(typeName) != null)
			return basePackage;
		var sourceFile = Path.Combine(binaryDir, typeName + Type.Extension);
		if (!File.Exists(sourceFile))
			return basePackage;
		//ncrunch: no coverage start
		if (!packagesByDirectory.TryGetValue(binaryDir, out var appPackage))
		{
			appPackage = new Package(basePackage, binaryDir);
			packagesByDirectory[binaryDir] = appPackage;
		}
		if (appPackage.FindDirectType(typeName) == null)
		{
			var typeLines = new TypeLines(typeName, File.ReadAllLines(sourceFile));
			_ = new Type(appPackage, typeLines).ParseMembersAndMethods(new MethodExpressionParser());
		}
		return appPackage; //ncrunch: no coverage end
	}

	private static IEnumerable<string> ReconstructSource(IList<Instruction> instructions)
	{
		var lines = new List<string>();
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
