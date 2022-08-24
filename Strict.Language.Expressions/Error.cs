using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Strict.Language.Expressions
{
	public sealed class Error : Expression
	{
		public Error(Expression message) : base(message.ReturnType) { }
	}
}
