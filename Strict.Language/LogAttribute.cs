#if DEBUG && LOGGING_ENABLED
using System.Diagnostics;
using System.Reflection;
using MethodDecorator.Fody.Interfaces;

namespace Strict.Language;

/// <summary>
/// If in DEBUG mode and FodyEnable is not false in the calling project (usually a test), then this
/// logs out methods marked with this so we can see all arguments and the order things are called.
/// In strict this will be a parser argument we can use so we can apply this to any type or project
/// </summary>
[Conditional("DEBUG")]
[AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor)]
public class LogAttribute : Attribute, IMethodDecorator
{
	public void Init(object instance, MethodBase method, object[] args)
	{
		if (instance is Body)
		{
			var type = instance.GetType();
			var methodValue = type.GetProperty("Method")?.GetValue(instance);
			var rangeValue = type.GetProperty("LineRange")?.GetValue(instance);
			Console.WriteLine($"[{nameof(Strict)}] Body.Parse {methodValue}, Lines={rangeValue}");
		}
		else
			Console.WriteLine($"[{
				nameof(Strict)
			}] {
				method.DeclaringType?.Name
			}{
				(method.Name == ".ctor"
					? ""
					: "." + method.Name)
			}({
				// ReSharper disable once ConditionIsAlwaysTrueOrFalseAccordingToNullableAPIContract
				string.Join(", ", args.Select(a => a != null
					? a.ToString()
					: "null"))
			})");
	}

	public void OnEntry() { }
	public void OnExit() { }
	public void OnException(Exception exception) { }
}
#else
namespace Strict.Language;

[AttributeUsage(AttributeTargets.Method | AttributeTargets.Constructor)]
public class LogAttribute : Attribute;
#endif