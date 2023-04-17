using System.Diagnostics;
using System.Reflection;

namespace Strict.Language;

public static class StackTraceExtensions
{
	/// <summary>
	/// Shows the callstack as multiline text output to help figure out who called what. Removes the
	/// first callstack line (this method) and all non-helpful System, NUnit and nCrunch lines.
	/// </summary>
	public static string FormatStackTraceIntoClickableMultilineText(int stackFramesToSkip = 0) =>
		EnhancedStackTrace.Current().GetFrames().Skip(stackFramesToSkip).
			Where(frame => !IsSystemOrTestMethodToExclude(frame.GetMethod())).Aggregate("",
				(current, frame) => current + "   at " + GetMethodWithParameters(frame.GetMethod()!) +
					GetFilenameAndLineInfo(frame) + "\n");

	private static bool IsSystemOrTestMethodToExclude(MemberInfo? method) =>
		method?.DeclaringType?.FullName?.StartsWith(MethodNamesToExclude) != false ||
		method.DeclaringType.Assembly.GetName().Name!.StartsWith("nCrunch", StringComparison.Ordinal);

	private static readonly string[] MethodNamesToExclude =
	{
		"DeltaEngine.Resolvers.", "DeltaEngine.Networking", "DeltaEngine.Content.ContentFileLoader",
		"DeltaEngine.Content.ContentDeserializer",
		"DeltaEngine.Mocks.Resolvers.TestWithMocksOrVisually", "System.Action",
		"System.RuntimeMethodHandle", "System.Reflection", "TestDriven", "System.Threading",
		"System.AppDomain", "System.Activator", "System.Runtime", "NUnit.Framework",
		"Microsoft.VisualStudio.HostingProcess", "System.Windows.", "System.Net.", "MS.Win32.",
		"MS.Internal.", "NUnit.Core.", "xUnit.", "JetBrains.ReSharper.", "nCrunch.", "Autofac.",
		"System.Reactive", "lambda_method", "System.Collections", "System.ThrowHelper",
		"System.Linq", "MongoDB"
	};

	private static string GetMethodWithParameters(MethodBase? method) =>
		method is null
			? string.Empty
			: method.DeclaringType + "." + method.Name + "(" + GetParameters(method) + ")";

	private static string GetParameters(MethodBase method) =>
		method.GetParameters().Aggregate("", (current, parameter) => current + (current.Length > 0
			? ", "
			: string.Empty) + parameter.ParameterType + " " + parameter.Name);

	private static string GetFilenameAndLineInfo(StackFrame frame)
	{
		var filename = frame.GetFileName();
		var lineNumber = frame.GetFileLineNumber();
		if (string.IsNullOrEmpty(filename) || lineNumber == 0)
			return string.Empty;
		return " in " + filename + ":line " + lineNumber;
	}
}