﻿using System.Globalization;

namespace Strict.Language.Expressions
{
	public class Number : Value
	{
		public Number(Context context, double value) :
			base(context.GetType(Base.Number), value) { }

		public override string ToString() =>
			((double)Data).ToString(CultureInfo.InvariantCulture);
	}
}