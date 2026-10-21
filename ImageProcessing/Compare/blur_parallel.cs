// blur.cs with the rows on all cores (Parallel.For), dotnet run -c Release blur_parallel.cs
using System.Diagnostics;
using System.Text;
var data = File.ReadAllBytes("Landscape.ppm");
var fields = Encoding.ASCII.GetString(data, 0, 40).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
int width = int.Parse(fields[1]), height = int.Parse(fields[2]);
int length = width * height * 3, stride = width * 3, radius = 4;
var pixels = data.AsSpan(data.Length - length).ToArray();
var blurred = new byte[length];
var area = (2 * radius + 1) * (2 * radius + 1);
var watch = Stopwatch.StartNew();
Parallel.For(0, height, y =>
{
	for (var x = 0; x < width; x++)
		for (var channel = 0; channel < 3; channel++)
		{
			var index = y * stride + x * 3 + channel;
			if (x < radius || x >= width - radius || y < radius || y >= height - radius)
				blurred[index] = pixels[index];
			else
			{
				var sum = 0;
				for (var dy = -radius; dy <= radius; dy++)
					for (var dx = -radius; dx <= radius; dx++)
						sum += pixels[index + dy * stride + dx * 3];
				blurred[index] = (byte)(sum / area);
			}
		}
});
Console.WriteLine($"C# Parallel.For: blurred {width}x{height} in {watch.Elapsed.TotalMilliseconds:F1} ms");
File.WriteAllBytes("Landscape.csharp_parallel.ppm", [.. Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n"), .. blurred]);
