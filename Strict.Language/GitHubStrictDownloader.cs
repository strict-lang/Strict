using System.Net.Http.Headers;
using System.Text.Json;

namespace Strict.Language;

public sealed class GitHubStrictDownloader : IDisposable
{
	public GitHubStrictDownloader(string owner, string repoNameAndFolders)
	{
		http = new HttpClient();
		http.DefaultRequestHeaders.UserAgent.Add(
			new ProductInfoHeaderValue(nameof(GitHubStrictDownloader), "1.0"));
		this.owner = owner;
		this.repoNameAndFolders = repoNameAndFolders;
	}

	private readonly HttpClient http;
	private readonly string owner;
	private readonly string repoNameAndFolders;

	public async Task DownloadFiles(string outputDirectory, CancellationToken token = default)
	{
		var apiUrl = "https://api.github.com/repos/" + owner + "/" + repoNameAndFolders +
			"/contents?ref=master";
		using var response = await http.GetAsync(apiUrl, token).ConfigureAwait(false);
		response.EnsureSuccessStatusCode();
		var json = await response.Content.ReadAsStringAsync(token).ConfigureAwait(false);
		var items = JsonSerializer.Deserialize<List<ContentItem>>(json,
			new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;
		foreach (var item in items)
			if (item.Type == "file" && item.Name is not null && item.DownloadUrl is not null &&
				item.Name.EndsWith(".strict", StringComparison.OrdinalIgnoreCase))
			{
				var localPath = Path.Combine(outputDirectory, item.Name);
				using var fileResponse = await http.GetAsync(item.DownloadUrl, token).ConfigureAwait(false);
				fileResponse.EnsureSuccessStatusCode();
				await using var fileStream = File.Create(localPath);
				await fileResponse.Content.CopyToAsync(fileStream, token).ConfigureAwait(false);
			}
	}

	private sealed class ContentItem
	{
		// ReSharper disable UnusedAutoPropertyAccessor.Local
		public string? Name { get; set; }
		public string? Type { get; set; }
		public string? DownloadUrl { get; set; }
	}

	public void Dispose() => http.Dispose();
}