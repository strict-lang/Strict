using DeltaEngine.VideoInputs;

namespace DeltaEngine.ImageProcessing.DepthFilters;

public class InitializeDepths : DepthImageProcessor
{
	public InitializeDepths(float initialDepth = ColorToDepthConverter.MinimumDistance) =>
		this.initialDepth = initialDepth;

	private readonly float initialDepth;

	public void Process(DepthImage depthImage)
	{
		for (var y = 0; y < depthImage.Height; y++)
		for (var x = 0; x < depthImage.Width; x++)
			depthImage.Depths[y * depthImage.Width + x] = initialDepth;
	}
}