using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Models;

namespace Faiss.Tests.Infrastructure;

public sealed record IndexCase(string Name, int Dimensions, MetricType Metric, Func<int, FloatIndex> Create, IndexCapabilities Capabilities) : ICapabilityCase
{
    public Func<FloatIndex, FloatIndex>? CloneIndex { get; init; }

    public Func<string, FloatIndex>? DeserializeFile { get; init; }

    public Func<Stream, FloatIndex>? Deserialize { get; init; }

    public int TrainCount { get; init; }

    public Func<IDSelector?, SearchParameters> CreateParameters { get; init; } = selector => new SearchParameters(selector);

    public Action<FloatIndex> EnableReconstruct { get; init; } = _ => { };

    public bool Has(IndexCapabilities caps) => Capabilities.HasFlag(caps);

    public FloatIndex CreateIndex()
    {
        var index = CreateUntrainedIndex();

        if (Has(IndexCapabilities.Train))
        {
            var train = Vectors.Random(seed: 98, count: TrainCount, dimensions: Dimensions);
            ((ITrainableFloatIndex)index).TrainAsync(TrainCount, train).GetAwaiter().GetResult();
        }

        return index;
    }

    public FloatIndex CreateUntrainedIndex() => Create(Dimensions);

    public override string ToString() => Name;
}
