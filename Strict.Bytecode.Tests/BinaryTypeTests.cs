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
    var source = new BinaryType
    {
      Members = [new BinaryMember("value", Type.Number, null)],
      MethodGroups = new Dictionary<string, List<BinaryType.BinaryMethod>>
      {
        ["Compute"] =
        [
          new BinaryType.BinaryMethod([new BinaryMember("input", Type.Number, null)],
            Type.Number, [new LoadConstantInstruction(Register.R0, Number(5)), new ReturnInstruction(Register.R0)])
        ]
      }
    };
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    source.Write(writer);
    writer.Flush();
    stream.Position = 0;
    using var reader = new BinaryReader(stream);
    var loaded = new BinaryType(reader, new BinaryExecutable(TestPackage.Instance), Type.Number);
    Assert.That(loaded.MethodGroups["Compute"][0].Instructions.Count, Is.EqualTo(2));
  }

  [Test]
  public void InvalidMagicThrows()
  {
    using var stream = new MemoryStream([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, BinaryType.Version]);
    using var reader = new BinaryReader(stream);
    Assert.That(() => new BinaryType(reader, new BinaryExecutable(TestPackage.Instance), Type.Number),
      Throws.TypeOf<BinaryType.InvalidBytecodeEntry>().With.Message.Contains("magic bytes"));
  }

  [Test]
  public void InvalidVersionThrows()
  {
    using var stream = new MemoryStream();
    using var writer = new BinaryWriter(stream);
    writer.Write("Strict"u8.ToArray());
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
    var method = new BinaryType.BinaryMethod([new BinaryMember("first", Type.Number, null)],
      Type.Number, [new ReturnInstruction(Register.R0)]);
    Assert.That(BinaryType.ReconstructMethodName("Compute", method),
      Is.EqualTo("Compute(first Number) Number"));
  }
}
