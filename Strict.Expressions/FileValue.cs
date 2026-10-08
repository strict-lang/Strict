using Type = Strict.Language.Type;

namespace Strict.Expressions;

public static class FileValue
{
	public static bool TryGetHandle(ValueInstance instance, Type fileType, out long handle)
	{
		if (instance.IsSameOrCanBeUsedAs(fileType) && !instance.IsText && !instance.IsList &&
			!instance.IsDictionary && instance.TryGetValueTypeInstance() == null)
		{
			handle = (long)instance.Number;
			return true;
		}
		handle = 0;
		return false;
	}

	/// <summary>
	/// Native I/O accepts Text or Path; Path (or bytecode-rebuilt types without Members) store their
	/// text as first value, so this is read positionally.
	/// </summary>
	public static bool TryGetPathText(ValueInstance pathValue, out string pathText)
	{
		if (pathValue.IsText)
		{
			pathText = pathValue.Text;
			return true;
		}
		var pathInstance = pathValue.TryGetValueTypeInstance();
		if (pathInstance is { Values: [{ IsText: true } textMember, ..] })
		{
			pathText = textMember.Text;
			return true;
		}
		pathText = "";
		return false;
	}

	public static ValueInstance CreateBytes(Type bytesType, Type byteType, byte[] bytes)
	{
		//TODO: horrible way to create byte array!
		var values = new ValueInstance[bytes.Length];
		for (var index = 0; index < bytes.Length; index++)
			values[index] = new ValueInstance(byteType, bytes[index]);
		return new ValueInstance(bytesType, values);
	}

	public static byte[] GetBytes(ValueInstance bytes)
	{
		var result = new byte[bytes.List.Items.Count];
		for (var index = 0; index < result.Length; index++)
			result[index] = (byte)Math.Clamp(bytes.List.Items[index].Number, 0, 255);
		return result;
	}
}