using System.Text;
using Strict.Bytecode;
using Strict.Bytecode.Instructions;
using Strict.Expressions;
using Strict.Language;
using Type = Strict.Language.Type;

namespace Strict.Compiler.Assembly;

public sealed partial class InstructionsToAssembly
{
	private static void EmitPrint(PrintInstruction print,
		List<(string Label, string Text)> printStrings, List<string> lines, Platform platform)
	{
		var (strLabel, _) = printStrings.First(p => p.Text == BuildPrintKey(print));
		if (print.ValueRegister.HasValue && !print.ValueIsText)
		{
			var numXmm = ToXmm(print.ValueRegister.Value);
			if (platform == Platform.Windows)
			{
				if (print.TextPrefix.Length > 0)
				{
					lines.Add("    sub rsp, 16");
					lines.Add("    movsd [rsp], " + numXmm);
					EmitWindowsWriteFromLabel(strLabel, print.TextPrefix.Length, lines);
					lines.Add("    movsd xmm0, [rsp]");
					lines.Add("    add rsp, 16");
					EmitWindowsWriteNumberFromXmm("xmm0", lines);
				}
				else
				{
					EmitWindowsWriteNumberFromXmm(numXmm, lines); //ncrunch: no coverage
				}
				return;
			}
			//ncrunch: no coverage start
			lines.Add($"    lea rdi, [rel {strLabel}]");
			if (numXmm != "xmm0")
				lines.Add($"    movsd xmm0, {numXmm}");
			lines.Add("    mov eax, 1");
			lines.Add("    call printf");
			return;
		}
		if (platform == Platform.Windows)
		{
			EmitWindowsWriteFromLabel(strLabel, BuildPrintKey(print).Length + 1, lines);
			return;
		}
		lines.Add($"    lea rdi, [rel {strLabel}]");
		lines.Add("    xor eax, eax");
		lines.Add("    call printf");
	} //ncrunch: no coverage end

	private static void EmitWindowsWriteFromLabel(string label, int length, List<string> lines)
	{
		if (length <= 0)
			return; //ncrunch: no coverage
		lines.Add("    sub rsp, 48");
		lines.Add("    mov ecx, -11");
		lines.Add("    call GetStdHandle");
		lines.Add("    mov rcx, rax");
		lines.Add("    lea rdx, [rel " + label + "]");
		lines.Add("    mov r8d, " + length);
		lines.Add("    lea r9, [rsp+40]");
		lines.Add("    mov qword [rsp+32], 0");
		lines.Add("    call WriteFile");
		lines.Add("    add rsp, 48");
	}

	private static void EmitWindowsWriteNumberFromXmm(string sourceXmm, List<string> lines)
	{
		if (sourceXmm != "xmm0")
			lines.Add("    movsd xmm0, " + sourceXmm); //ncrunch: no coverage
		lines.Add("    call print_number_from_xmm");
	}

	private static string BuildWindowsPrintNumberHelper() =>
		string.Join("\n", "", "section .text", "print_number_from_xmm:", "    push rbp",
			"    mov rbp, rsp", "    sub rsp, 96", "    movsd [rsp], xmm0", "    mov ecx, -11",
			"    call GetStdHandle", "    mov rcx, rax", "    lea r10, [rsp+79]",
			"    mov byte [r10], 10", "    mov r11, r10", "    movsd xmm0, [rsp]",
			"    cvttsd2si rax, xmm0", "    xor r9d, r9d", "    test rax, rax", "    jge .print_abs_done",
			"    mov r9d, 1", "    neg rax", ".print_abs_done:", "    test rax, rax",
			"    jne .print_digits_loop", "    dec r11", "    mov byte [r11], '0'",
			"    jmp .print_digits_done", ".print_digits_loop:", "    xor edx, edx", "    mov r8, 10",
			"    div r8", "    add dl, '0'", "    dec r11", "    mov [r11], dl", "    test rax, rax",
			"    jne .print_digits_loop", ".print_digits_done:", "    test r9d, r9d",
			"    je .print_sign_done", "    dec r11", "    mov byte [r11], '-'", ".print_sign_done:",
			"    mov rdx, r11", "    mov r8, r10", "    sub r8, r11", "    inc r8",
			"    lea r9, [rsp+40]", "    mov qword [rsp+32], 0", "    call WriteFile", "    add rsp, 96",
			"    pop rbp", "    ret");

	private static string BuildPrintKey(PrintInstruction print) =>
		print.ValueRegister.HasValue && !print.ValueIsText
			? print.TextPrefix + "%g"
			: print.TextPrefix;

	private static List<(string Label, string Text)> CollectPrintStrings(
		List<Instruction> instructions)
	{
		var strings = new List<(string, string)>();
		var seen = new HashSet<string>(StringComparer.Ordinal);
		var labelIndex = 0;
		for (var instructionIndex = 0; instructionIndex < instructions.Count; instructionIndex++)
		{
			if (instructions[instructionIndex].InstructionType != InstructionType.Print)
				continue;
			var instruction = (PrintInstruction)instructions[instructionIndex];
			var key = BuildPrintKey(instruction);
			if (seen.Add(key))
				strings.Add(($"str_{labelIndex++}", key));
		}
		return strings;
	}

	private static string BuildStringBytes(string text)
	{
		if (text.Length == 0)
			return ""; //ncrunch: no coverage
		var parts = new List<string>();
		var ascii = new StringBuilder();
		foreach (var c in text)
			if (c is >= ' ' and <= '~' && c != '"' && c != '\\')
			{
				ascii.Append(c);
			}
			else
			{ //ncrunch: no coverage start
				if (ascii.Length > 0)
				{
					parts.Add($"\"{ascii}\"");
					ascii.Clear();
				}
				parts.Add(((int)c).ToString());
			} //ncrunch: no coverage end
		if (ascii.Length > 0)
			parts.Add($"\"{ascii}\"");
		return string.Join(", ", parts);
	}
}
