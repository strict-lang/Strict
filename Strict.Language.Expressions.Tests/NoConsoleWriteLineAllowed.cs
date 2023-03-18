using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using NUnit.Framework;
using NUnit.Framework.Interfaces;

namespace Strict.Language.Expressions.Tests;

public class NoConsoleWriteLineAllowed
{
	[SetUp]
	public void SetVirtualConsoleWriter() => Console.SetOut(ConsoleWriter);

	public static readonly VirtualConsoleWriter ConsoleWriter = new(Console.Out);

	public sealed class VirtualConsoleWriter : TextWriter
	{
		public VirtualConsoleWriter(TextWriter originalOutput) => this.originalOutput = originalOutput;
		private readonly TextWriter originalOutput;
		public override Encoding Encoding => Encoding.UTF8; //ncrunch: no coverage

		public override void Write(string? value)
		{
			originalOutput.Write(value);
			if (string.IsNullOrEmpty(value) || IsCategoryManual() ||
				value.Contains("Message was logged and copied to the clipboard"))
				return; //ncrunch: no coverage
			rememberTextWritten += value + Environment.NewLine;
		}

		public static bool IsCategoryManual()
		{
			foreach (var frame in new StackTrace().GetFrames())
				if (HasAttribute(frame, TestAttribute) &&
					HasAttribute(frame, NunitFrameworkCategoryAttribute))
					return GetCategoryName(frame) == ManualCategoryName;
			return false;
		}

		public static bool HasAttribute(StackFrame frame, string name)
		{
			var attributes = frame.GetMethod()!.GetCustomAttributes(false);
			return attributes.Any(attribute => attribute.GetType().ToString() == name);
		}

		private const string TestAttribute = "NUnit.Framework.TestAttribute";
		private const string NunitFrameworkCategoryAttribute = "NUnit.Framework.CategoryAttribute";

		private static string GetCategoryName(StackFrame frame)
		{
			var method = frame.GetMethod();
			var attributes = method?.GetCustomAttributes(false);
			var attribute = attributes?.FirstOrDefault(a =>
				a.GetType().ToString() == NunitFrameworkCategoryAttribute);
			return attribute != null
				? (string)attribute.GetType().GetProperty("Name")!.GetValue(attribute, null)!
				: string.Empty;
		}

		private const string ManualCategoryName = "Manual";
		private string rememberTextWritten = string.Empty;
		public override void WriteLine(string? value) => Write(value + Environment.NewLine);
		public void Clear() => Dispose();

		protected override void Dispose(bool disposing)
		{
			base.Dispose(disposing);
			Console.SetOut(originalOutput);
			rememberTextWritten = string.Empty;
		}

		public bool IsEmpty => rememberTextWritten.Length == 0;

		public string GetTextAndClear()
		{
			var text = rememberTextWritten;
			Clear();
			return text;
		}
	}

	/// <summary>
	/// Reports if there is something in the console and our test is not failing. Also checks if
	/// another test might have just failed and ignores it if console still have that output around.
	/// </summary>
	[TearDown]
	public void CheckIfConsoleIsEmpty()
	{
		if (ConsoleWriter.IsEmpty ||
			TestContext.CurrentContext.Result.Outcome.Status is TestStatus.Failed)
			return;
		var textInConsole = ConsoleWriter.GetTextAndClear();
		if (!textInConsole.StartsWith("  Expected: ", StringComparison.Ordinal))
			throw new ConsoleWriteLineShouldOnlyBeUsedInManualTests(textInConsole);
	}

	public sealed class ConsoleWriteLineShouldOnlyBeUsedInManualTests : Exception
	{
		public ConsoleWriteLineShouldOnlyBeUsedInManualTests(string message) : base(message) { }
	}
}