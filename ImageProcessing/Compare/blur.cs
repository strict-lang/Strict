// Same blur as BenchBlur.strict and blur.c, usage: dotnet run -c Release blur.cs width height radius
using System.Diagnostics;
var numbers = args.Select(int.Parse).Concat(new[] { 3840, 2160, 4 }.Skip(args.Length)).ToArray();
int width = numbers[0], height = numbers[1], radius = numbers[2];
int length = width * height * 4, stride = width * 4;
var pixels = new byte[length];
for (var index = 0; index < length; index++)
	pixels[index] = (byte)((long)index * 7919 % 256);
var blurred = new byte[length];
var area = (2 * radius + 1) * (2 * radius + 1);
var watch = Stopwatch.StartNew();
for (var y = 0; y < height; y++)
	for (var x = 0; x < width; x++)
		for (var channel = 0; channel < 4; channel++)
		{
			var index = y * stride + x * 4 + channel;
			if (x < radius || x >= width - radius || y < radius || y >= height - radius)
				blurred[index] = pixels[index];
			else
			{
				var sum = 0;
				for (var dy = -radius; dy <= radius; dy++)
					for (var dx = -radius; dx <= radius; dx++)
						sum += pixels[index + dy * stride + dx * 4];
				blurred[index] = (byte)(sum / area);
			}
		}
Console.WriteLine($"C#: blurred {width}x{height} with {2 * radius + 1}x{2 * radius + 1} in {watch.Elapsed.TotalMilliseconds:F1} ms, pixel {blurred[length / 2 + stride / 2]}");
