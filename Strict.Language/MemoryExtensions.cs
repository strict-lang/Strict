using System;

namespace Strict.Language
{
	public static class MemoryExtensions2
	{
		public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span) =>
			new SpanSplitEnumerator<char>(span, ' ');

		public static SpanSplitEnumerator<char> Split(this ReadOnlySpan<char> span, char separator) =>
			new SpanSplitEnumerator<char>(span, separator);

		public static SpanSplitSequenceEnumerator<char> Split(this ReadOnlySpan<char> span,
			string separator) =>
			new SpanSplitSequenceEnumerator<char>(span, separator);
	}

	public ref struct SpanSplitEnumerator<T> where T : IEquatable<T>
	{
		private readonly ReadOnlySpan<T> _sequence;
		private readonly T _separator;
		private int _offset;
		private int _index;
		public SpanSplitEnumerator<T> GetEnumerator() => this;

		internal SpanSplitEnumerator(ReadOnlySpan<T> span, T separator)
		{
			_sequence = span;
			_separator = separator;
			_index = 0;
			_offset = 0;
		}

		public Range Current => new Range(_offset, _offset + _index - 1);

		public bool MoveNext()
		{
			if (_sequence.Length - _offset < _index)
				return false;
			var slice = _sequence.Slice(_offset += _index);
			var nextIdx = slice.IndexOf(_separator);
			_index = (nextIdx != -1
				? nextIdx
				: slice.Length) + 1;
			return true;
		}
	}

	public ref struct SpanSplitSequenceEnumerator<T> where T : IEquatable<T>
	{
		private readonly ReadOnlySpan<T> _sequence;
		private readonly ReadOnlySpan<T> _separator;
		private int _offset;
		private int _index;
		public SpanSplitSequenceEnumerator<T> GetEnumerator() => this;

		internal SpanSplitSequenceEnumerator(ReadOnlySpan<T> span, ReadOnlySpan<T> separator)
		{
			_sequence = span;
			_separator = separator;
			_index = 0;
			_offset = 0;
		}

		public Range Current => new Range(_offset, _offset + _index - 1);

		public bool MoveNext()
		{
			if (_sequence.Length - _offset < _index)
				return false;
			var slice = _sequence.Slice(_offset += _index);
			var nextIdx = slice.IndexOf(_separator);
			_index = (nextIdx != -1
				? nextIdx
				: slice.Length) + 1;
			return true;
		}
	}

	//public static class SpanSplitTests
	//{
	//    [Fact]
	//    public static void SplitNoMatchSingleResult()
	//    {
	//        ReadOnlySpan<char> value = "a b";

	//        string expected = value.ToString();
	//        var enumerator = value.Split(',');
	//        Assert.True(enumerator.MoveNext());
	//        Assert.Equal(expected,  value.Slice(enumerator.Current).ToString());
	//    }

	//    [Theory]
	//    [InlineData("", ',', new [] { "" })]
	//    [InlineData(",", ',', new [] { "", "" })]
	//    [InlineData(",,", ',', new [] { "", "", "" })]
	//    [InlineData("ab", ',', new [] { "ab" })]
	//    [InlineData("a,b", ',', new [] { "a", "b" })]
	//    [InlineData("a,", ',', new [] { "a", "" })]
	//    [InlineData(",b", ',', new [] { "", "b" })]
	//    [InlineData(",a,b", ',', new [] { "", "a", "b" })]
	//    [InlineData("a,b,", ',', new [] { "a", "b", "" })]
	//    [InlineData("a,b,c", ',', new [] { "a", "b", "c" })]
	//    [InlineData("a,,c", ',', new [] { "a", "", "c" })]
	//    [InlineData(",a,b,c", ',', new [] { "", "a", "b", "c" })]
	//    [InlineData("a,b,c,", ',', new [] { "a", "b", "c", "" })]
	//    [InlineData(",a,b,c,", ',', new [] { "", "a", "b", "c", "" })]
	//    [InlineData("first,second", ',', new [] { "first", "second" })]
	//    [InlineData("first,", ',', new [] { "first", "" })]
	//    [InlineData(",second", ',', new [] { "", "second" })]
	//    [InlineData(",first,second", ',', new [] { "", "first", "second" })]
	//    [InlineData("first,second,", ',', new [] { "first", "second", "" })]
	//    [InlineData("first,second,third", ',', new [] { "first", "second", "third" })]
	//    [InlineData("first,,third", ',', new [] { "first", "", "third" })]
	//    [InlineData(",first,second,third", ',', new [] { "", "first", "second", "third" })]
	//    [InlineData("first,second,third,", ',', new [] { "first", "second", "third", "" })]
	//    [InlineData(",first,second,third,", ',', new [] { "", "first", "second", "third", "" })]
	//    [InlineData("Foo Bar Baz", ' ', new[] { "Foo", "Bar", "Baz" })]
	//    [InlineData("Foo Bar Baz ", ' ', new[] { "Foo", "Bar", "Baz", "" })]
	//    [InlineData(" Foo Bar Baz ", ' ', new[] { "", "Foo", "Bar", "Baz", "" })]
	//    [InlineData(" Foo  Bar Baz ", ' ', new[] { "", "Foo", "", "Bar", "Baz", "" })]
	//    public static void SplitCharSeparator(string valueParam, char separator, string[] expectedParam)
	//    {
	//        ReadOnlySpan<char> value = valueParam;
	//        char[][] expected = expectedParam.Select(x => x.ToCharArray()).ToArray();
	//        SplitHelpers.AssertEqual(valueParam, value.Split(separator), expected);
	//        SplitHelpers.AssertEqual(valueParam, value.Split(separator.ToString()), expected);
	//    }

	//    private static class SplitHelpers
	//    {
	//        public static void AssertEqual<T>(ReadOnlySpan<T> orig, SpanSplitEnumerator<T> source, T[][] items) where T : IEquatable<T>
	//        {
	//            foreach (var item in items)
	//            {
	//                Assert.True(source.MoveNext());
	//                var slice = orig.Slice(source.Current);
	//                Assert.Equal(item.Length, slice.Length);
	//                for (int idx = 0; idx < item.Length; idx++)
	//                {
	//                    Assert.Equal(item[idx], slice[idx]);
	//                }
	//            }
	//            Assert.False(source.MoveNext());
	//        }

	//        public static void AssertEqual<T>(ReadOnlySpan<T> orig, SpanSplitSequenceEnumerator<T> source, T[][] items) where T : IEquatable<T>
	//        {
	//            foreach (var item in items)
	//            {
	//                Assert.True(source.MoveNext());
	//                var slice = orig.Slice(source.Current);
	//                Assert.Equal(item.Length, slice.Length);

	//                for (int idx = 0; idx < item.Length; idx++)
	//                {
	//                    Assert.Equal(item[idx], slice[idx]);
	//                }
	//            }
	//            Assert.False(source.MoveNext());
	//        }
	//    }
	//}
}