using System.Text.Json.Serialization;
using Faiss.Server.Models;

namespace Faiss.Server;

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower)]
[JsonSerializable(typeof(CreateIndexRequest))]
[JsonSerializable(typeof(IReadOnlyList<IndexInfo>))]
internal partial class AppJsonContext : JsonSerializerContext;