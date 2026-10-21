// blur.cs with the rows on all cores (Parallel.For), dotnet run -c Release blur_parallel.cs
using System.Diagnostics;
using System.Text;
var file = File.ReadAllBytes("Landscape.ppm");
var fields = Encoding.ASCII.GetString(file, 0, 40).Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
int width = int.Parse(fields[1]), height = int.Parse(fields[2]);
var rgb = file[^(width * height * 3)..];
var data = new byte[width * height * 4];
for (var pixel = 0; pixel < width * height; ++pixel)
	(data[pixel * 4], data[pixel * 4 + 1], data[pixel * 4 + 2], data[pixel * 4 + 3]) = (rgb[pixel * 3], rgb[pixel * 3 + 1], rgb[pixel * 3 + 2], 255);
var blurred = new byte[width * height * 4];
var watch = Stopwatch.StartNew();
Parallel.For(0, height, y =>
{
	for (var x = 0; x < width; ++x)
	{
		int red = 0, green = 0, blue = 0, alpha = 0, count = 0;
		for (var dy = -4; dy <= 4; ++dy)
			for (var dx = -4; dx <= 4; ++dx)
				if (x + dx >= 0 && x + dx < width && y + dy >= 0 && y + dy < height)
				{
					var sourceIndex = ((y + dy) * width + x + dx) * 4;
					red += data[sourceIndex];
					green += data[sourceIndex + 1];
					blue += data[sourceIndex + 2];
					alpha += data[sourceIndex + 3];
					++count;
				}
		var targetIndex = (y * width + x) * 4;
		blurred[targetIndex] = (byte)(red / count);
		blurred[targetIndex + 1] = (byte)(green / count);
		blurred[targetIndex + 2] = (byte)(blue / count);
		blurred[targetIndex + 3] = (byte)(alpha / count);
	}
});
Console.WriteLine($"C# Parallel.For: blurred {width}x{height} in {watch.Elapsed.TotalMilliseconds:F1} ms");
File.WriteAllBytes("Landscape.csharp_parallel.ppm", [.. Encoding.ASCII.GetBytes($"P6\n{width} {height}\n255\n"), .. blurred.Where((_, index) => index % 4 != 3)]);
