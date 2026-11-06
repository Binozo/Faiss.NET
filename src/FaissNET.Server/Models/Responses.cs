using Faiss.Models;

namespace Faiss.Server.Models;

public record IndexInfo
{
    public required Guid Id { get; init; }
    public required string FactoryString { get; init; }
    public required int Dimension { get; init; }
    public required MetricType MetricType { get; init; }
    public required long NTotal { get; init; }
    public required bool IsTrained  { get; init; }
    public required DateTimeOffset CreatedAt { get; init; }
    public required DateTimeOffset UpdatedAt { get; init; }
}