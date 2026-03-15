using System.Text;
using Strict.Bytecode.Instructions;

namespace Strict.Bytecode;

/// <summary>
/// Partially reconstructs .strict source files from BytecodeTypes (e.g. from .strictbinary) as an
/// approximation. For debugging, will not compile, no tests. Only includes what bytecode reveals.
/// </summary>
public sealed class BytecodeDecompiler
{
	/// <summary>
	/// Opens a .strictbinary ZIP file, deserializes each bytecode entry, and writes
	/// a reconstructed .strict source file per entry into <paramref name="outputFolder" />.
	/// </summary>
	public void Decompile(BytecodeTypes allInstructions, string outputFolder)
	{
		Directory.CreateDirectory(outputFolder);
		foreach (var typeMethods in allInstructions.MethodsPerType)
		{
			var sourceLines = ReconstructSource(typeMethods.Value);
			var outputPath = Path.Combine(outputFolder, typeMethods.Key + ".strict");
			File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
		}
			/*obs
		var binaryDir = Path.GetDirectoryName(Path.GetFullPath(strictBinaryFilePath)) ?? ".";
		using var zip = ZipFile.OpenRead(strictBinaryFilePath);
		foreach (var entry in zip.Entries)
			if (entry.Name.EndsWith(".bytecode", StringComparison.OrdinalIgnoreCase))
			{
				var typeName = Path.GetFileNameWithoutExtension(entry.Name);
				using var entryStream = entry.Open();
				var instructions = BytecodeDeserializer.DeserializeEntry(entryStream,
					GetPackageForType(binaryDir, typeName));
				var sourceLines = ReconstructSource(instructions);
				var outputPath = Path.Combine(outputFolder, typeName + ".strict");
				File.WriteAllLines(outputPath, sourceLines, Encoding.UTF8);
			}
			*/
	}

	/*
		private readonly Dictionary<string, Package> packagesByDirectory = new();

		private Package GetPackageForType(string binaryDir, string typeName)
		{
			if (basePackage.FindType(typeName) != null)
				return basePackage;
			//ncrunch: no coverage start
			var sourceFile = Path.Combine(binaryDir, typeName + Type.Extension);
			if (!File.Exists(sourceFile))
				return basePackage;
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
	*/
	private static IReadOnlyList<string> ReconstructSource(BytecodeTypes.TypeMembersAndMethods typeData)
	{
		var lines = new List<string>();
		foreach (var member in typeData.Members)
			lines.Add("has " + member.Name + " " + member.FullTypeName +
				(member.InitialValueExpression != null
					? " = " + member.InitialValueExpression
					: ""));
		foreach (var (methodKey, instructions) in typeData.InstructionsPerMethod)
		{
			lines.Add(methodKey); //TODO: this is not real strict code, we should reconstruct better!
			var bodyLines = new List<string>();
			for (var index = 0; index < instructions.Count; index++)
			{
				switch (instructions[index])
				{
				case StoreVariableInstruction storeVar:
					//ncrunch: no coverage start
					bodyLines.Add("\tconstant " + storeVar.Identifier + " = " +
						storeVar.ValueInstance.ToExpressionCodeString());
					break; //ncrunch: no coverage end
				case Invoke invoke when invoke.Method != null &&
					index + 1 < instructions.Count &&
					instructions[index + 1] is StoreFromRegisterInstruction nextStore &&
					nextStore.Register == invoke.Register:
					bodyLines.Add("\tconstant " + nextStore.Identifier + " = " + invoke.Method);
					index++;
					break;
				case Invoke { Method: not null } invoke:
					//ncrunch: no coverage start
					bodyLines.Add("\t" + invoke.Method);
					break;
				} //ncrunch: no coverage end
			}
			lines.AddRange(bodyLines);
		}
		/*TODO: cleanup,  what nonsense is this? the above is already wrong, but this is just madness
		var members = new List<string>();
		var bodyLines = new List<string>();
		for (var index = 0; index < instructions.Count; index++)
		{
			switch (instructions[index])
			{
			case StoreVariableInstruction storeVar when storeVar.IsMember:
				members.Add("has " + storeVar.Identifier + " " +
					storeVar.ValueInstance.GetType().Name);
				//TODO:  " = " + storeVar.ValueInstance.ToExpressionCodeString());
				break;
			case StoreVariableInstruction storeVar:
				//ncrunch: no coverage start
				bodyLines.Add("\tconstant " + storeVar.Identifier + " = " +
					storeVar.ValueInstance.ToExpressionCodeString());
				break; //ncrunch: no coverage end
			case Invoke invoke when invoke.Method != null &&
				index + 1 < instructions.Count &&
				instructions[index + 1] is StoreFromRegisterInstruction nextStore &&
				nextStore.Register == invoke.Register:
				bodyLines.Add("\tconstant " + nextStore.Identifier + " = " + invoke.Method);
				index++;
				break;
			case Invoke { Method: not null } invoke:
				//ncrunch: no coverage start
				bodyLines.Add("\t" + invoke.Method);
				break;
			} //ncrunch: no coverage end
		}
		var lines = new List<string>(members);
		if (bodyLines.Count > 0)
		{
			lines.Add("Run");
			lines.AddRange(bodyLines);
		}
		*/
		return lines;
	}
}