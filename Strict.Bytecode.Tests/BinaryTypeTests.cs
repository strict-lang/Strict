using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Bytecode.Tests;

public sealed class BinaryTypeTests : TestBytecode
{
  [Test]
  public void WriteAndReadPreservesMethodInstructions()
  {
		var binary = new BinaryExecutable(TestPackage.Instance);
		var source = new BinaryType(binary, nameof(BinaryTypeTests),
			[new BinaryMember("value", Type.Number, null)],
			new Dictionary<string, List<BinaryMethod>>
			{
				["Compute"] =
				[
					new BinaryMethod("", [new BinaryMember("input", Type.Number, null)],
						Type.Number, [new LoadConstantInstruction(Register.R0, Number(5)),
						new ReturnInstruction(Register.R0)])
				]
			});
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    source.Write(writer);
    writer.Flush();
    stream.Position = 0;
    using var reader = new BinaryReader(stream);
    var loaded = new BinaryType(reader, binary, nameof(BinaryTypeTests));
    Assert.That(loaded.MethodGroups["Compute"][0].instructions.Count, Is.EqualTo(2));
  }

  [Test]
  public void InvalidMagicThrows()
  {
    using var stream = new MemoryStream([0x01, BinaryType.Version]);
    using var reader = new BinaryReader(stream);
    Assert.That(() => new BinaryType(reader, new BinaryExecutable(TestPackage.Instance), Type.Number),
      Throws.TypeOf<BinaryType.InvalidBytecodeEntry>().With.Message.Contains("magic byte"));
  }

  [Test]
  public void InvalidVersionThrows()
  {
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    writer.Write((byte)'S');
    writer.Write((byte)0);
    writer.Flush();
    stream.Position = 0;
    using var reader = new BinaryReader(stream);
    Assert.That(() => new BinaryType(reader, new BinaryExecutable(TestPackage.Instance), Type.Number),
      Throws.TypeOf<BinaryType.InvalidVersion>());
  }

  [Test]
  public void ReconstructMethodNameIncludesParametersAndReturnType()
  {
    var method = new BinaryMethod("", [new BinaryMember("first", Type.Number, null)],
      Type.Number, [new ReturnInstruction(Register.R0)]);
    Assert.That(BinaryType.ReconstructMethodName("Compute", method),
      Is.EqualTo("Compute(first Number) Number"));
  }
}