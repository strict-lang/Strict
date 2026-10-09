using System.Globalization;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Bytecode.Serialization;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict;

public sealed partial class VirtualMachine
{
	private bool TryHandleNativeStaticTypeMethod(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		return info.ShortTypeName switch
		{
			"Process" => TryHandleProcessStatic(invoke),
			"Directory" => TryHandleDirectoryStatic(invoke),
			_ => false
		};
	}

	private bool TryHandleProcessStatic(Invoke invoke)
	{
		var info = invoke.MethodInfo;
		if (info.MethodName == "Find" && info.ArgumentRegisters.Length == 1)
		{
			var name = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			var path = NativeProcessRunner.FindTool(name) ?? "";
			Memory.Registers[invoke.Register] = CreateProcessValue(path);
			return true;
		}
		if (info.MethodName == NativeProcessRunner.OperatingSystemMethod)
		{
			Memory.Registers[invoke.Register] = new ValueInstance(NativeProcessRunner.OperatingSystemName);
			return true;
		}
		if (info.MethodName == "RunTool" && info.ArgumentRegisters.Length == 2)
		{
			var name = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			var arguments = GetArgumentText(Memory.Registers[info.ArgumentRegisters[1]]);
			var path = NativeProcessRunner.FindTool(name);
			Memory.Registers[invoke.Register] = path == null
				? CreateProcessResultValue(127, "", "tool not found: " + name)
				: CreateProcessResultValue(NativeProcessRunner.Run(path, arguments));
			return true;
		}
		return false;
	}

	private bool TryHandleNativeProcessInstanceMethod(Invoke invoke, ValueInstance? implicitInstance)
	{
		var info = invoke.MethodInfo;
		if (info.ShortTypeName != "Process" &&
			!IsProcessInstance(ResolveInvokeInstanceSafe(info, implicitInstance)))
			return false;
		if (info.MethodName == "IsAvailable")
		{
			var instance = ResolveInvokeInstance(info, implicitInstance);
			Memory.Registers[invoke.Register] = new ValueInstance(executable.booleanType,
				GetProcessExecutable(instance).Length > 0);
			return true;
		}
		if (info.MethodName == "Run" && info.ArgumentRegisters.Length == 1)
		{
			var instance = ResolveInvokeInstance(info, implicitInstance);
			var executablePath = GetProcessExecutable(instance);
			var arguments = GetArgumentText(Memory.Registers[info.ArgumentRegisters[0]]);
			Memory.Registers[invoke.Register] = executablePath.Length == 0
				? CreateProcessResultValue(127, "", "Process has no executable")
				: CreateProcessResultValue(NativeProcessRunner.Run(executablePath, arguments));
			return true;
		}
		return false;
	}

	private ValueInstance? ResolveInvokeInstanceSafe(InvokeMethodInfo info,
		ValueInstance? implicitInstance)
	{
		if (info.InstanceRegister.HasValue)
			return Memory.Registers[info.InstanceRegister.Value];
		return implicitInstance;
	}

	private bool IsProcessInstance(ValueInstance? instance)
	{
		if (instance is not { HasValue: true, IsFlatNumeric: false })
			return false;
		var typeInstance = instance.Value.TryGetValueTypeInstance();
		return typeInstance != null && typeInstance.ReturnType.Name == "Process";
	}

	private static string GetProcessExecutable(ValueInstance instance)
	{
		var typeInstance = instance.TryGetValueTypeInstance();
		if (typeInstance != null && typeInstance.TryGetValue("executable", out var executableValue) &&
			executableValue.IsText)
			return executableValue.Text;
		return "";
	}

	private ValueInstance CreateProcessValue(string executablePath)
	{
		var processType = executable.TypeResolver.GetType("Process");
		return new ValueInstance(processType, [new ValueInstance(executablePath)]);
	}

	private ValueInstance CreateProcessResultValue(NativeProcessRunner.ProcessRunResult result) =>
		CreateProcessResultValue(result.ExitCode, result.Output, result.Error);

	private ValueInstance CreateProcessResultValue(int exitCode, string output, string error)
	{
		var resultType = executable.TypeResolver.GetType("ProcessResult");
		var numberType = executable.numberType;
		return new ValueInstance(resultType, [
			new ValueInstance(numberType, exitCode),
			new ValueInstance(output),
			new ValueInstance(error)
		]);
	}
}
