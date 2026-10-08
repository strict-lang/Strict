using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.HighLevelRuntime;

/// <summary>
/// Directory.strict methods have no body, they are executed natively like in the VirtualMachine.
/// </summary>
internal sealed class DirectoryEvaluator(Interpreter interpreter)
{
	public bool TryEvaluate(Method method, IReadOnlyList<ValueInstance> args,
		out ValueInstance result)
	{
		result = interpreter.noneInstance;
		if (method.Type.Name != Type.Directory || args.Count == 0 ||
			!FileValue.TryGetPathText(args[0], out var path))
			return false;
		switch (method.Name)
		{
		case "Exists":
			result = interpreter.ToBoolean(NativeDirectory.Exists(path));
			return true;
		case "Create":
			NativeDirectory.Create(path);
			return true;
		case "Files" or "GetFiles":
			result = interpreter.CreateTexts(method, NativeDirectory.GetFiles(path, args.Count > 1 &&
				FileValue.TryGetPathText(args[1], out var pattern)
					? pattern
					: ""));
			return true;
		default:
			return false;
		}
	}
}
