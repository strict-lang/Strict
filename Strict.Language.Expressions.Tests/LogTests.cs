using System;
using NUnit.Framework;
using Strict.Language.Tests;

namespace Strict.Language.Expressions.Tests
{
	public class LogTests
	{
		[Test]
		public void PrintHelloWorld()
		{
			var package = new TestPackage();
			new Type(package, Base.App, null).Parse("Run");
			var type = new Type(package, "Program", null).Parse(@"implement App
has log
Run
	log.Write(text+number)");
			// var interpreter = new Interpreter();
			// interpreter.Run(type.Methods[0]);
			Assert.That(type, Is.Not.Null);
		}

		//ncrunch: no coverage start
		public class Interpreter
		{
			public void Run(Method method)
			{
				foreach (var expression in method.Body.Expressions)
					if (expression is not MethodCall)
						throw new NotSupportedException();
					else
					{
						var call = (MethodCall)expression;
						if (call.Method.Name == "Write")
							Console.WriteLine((call.Arguments[0] as Text).Data);
						else
							throw new NotSupportedException();
					}
			}
		}
	}
}