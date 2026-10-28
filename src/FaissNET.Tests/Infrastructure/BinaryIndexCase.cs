using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Models;

namespace Faiss.Tests.Infrastructure;

public sealed record BinaryIndexCase(string Name, int Dimensions, MetricType Metric, Func<int, BinaryIndex> Create, IndexCapabilities Capabilities) : ICapabilityCase
{
    public int CodeSize => Dimensions / 8;

    public Func<BinaryIndex, BinaryIndex>? CloneIndex { get; init; }

    public Func<string, BinaryIndex>? DeserializeFile { get; init; }

    public Func<Stream, BinaryIndex>? Deserialize { get; init; }

    /// <summary>Training vectors required before this index accepts adds.</summary>
    public int TrainCount { get; init; }

    public Func<IDSelector?, SearchParameters> CreateParameters { get; init; } = selector => new SearchParameters(selector);

    public bool Has(IndexCapabilities caps) => Capabilities.HasFlag(caps);

    public BinaryIndex CreateIndex()
    {
        var index = Create(Dimensions);

        if (Has(IndexCapabilities.Train))
        {
            var train = Vectors.RandomBinary(seed: 99, count: TrainCount, bitDimensions: Dimensions);
            ((ITrainableBinaryIndex)index).TrainAsync(TrainCount, train).GetAwaiter().GetResult();
        }

        return index;
    }

    public override string ToString() => Name;
}
