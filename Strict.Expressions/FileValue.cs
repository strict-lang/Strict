using Type = Strict.Language.Type;

namespace Strict.Expressions;

public static class FileValue
{
	public static bool TryGetPath(ValueInstance instance, out string path)
	{
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null && typeInstance.TryGetValue("path", out var pathValue) &&
			pathValue.IsText)
		{
			path = pathValue.Text;
			return true;
		}
		path = "";
		return false;
	}

	public static ValueInstance CreateBytes(Type bytesType, Type byteType, byte[] bytes)
	{
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
