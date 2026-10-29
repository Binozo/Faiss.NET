using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Models;

public class EnumMappingTests
{
    private const int Dimensions = 4;

    [Theory]
    [InlineData(MetricType.L2)]
    [InlineData(MetricType.InnerProduct)]
    [InlineData(MetricType.L1)]
    [InlineData(MetricType.Linf)]
    [InlineData(MetricType.Canberra)]
    [InlineData(MetricType.BrayCurtis)]
    [InlineData(MetricType.JensenShannon)]
    public void MetricType_RoundTripsThroughTheFactory(MetricType metric)
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("Flat", Dimensions, metric);

        Assert.Equal(metric, index.Metric);
    }

    [Theory]
    [InlineData(MetricType.L1)]
    [InlineData(MetricType.Canberra)]
    public void MetricType_NonDefaultMetrics_AreActuallyUsedForSearch(MetricType metric)
    {
        using var index = IndexFactory.Create<GenericFloatIndex>("Flat", Dimensions, metric);

        var vectors = Vectors.Random(seed: 220, count: 8, dimensions: Dimensions);
        index.Add(8, vectors);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, Dimensions), 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Fact]
    public void QuantizerTrainMode_DefaultsToKmeansOnTheQuantizerIndex()
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, nlist: 2);

        Assert.Equal(QuantizerTrainMode.QuantizerIndexKmeansTraining, index.QuantizerTrainMode);
        Assert.Equal(0, (sbyte)QuantizerTrainMode.QuantizerIndexKmeansTraining);
    }
}
