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

    [Theory]
    [InlineData(RangeStat.RS_minmax, "RS_minmax")]
    [InlineData(RangeStat.RS_meanstd, "RS_meanstd")]
    [InlineData(RangeStat.RS_quantiles, "RS_quantiles")]
    [InlineData(RangeStat.RS_optim, "RS_optim")]
    public void RangeStat_ValuesMatchTheNativeEnumeration(RangeStat managed, string member)
    {
        // No exposing in the c_api thus we need to verify the source ourselves
        Assert.Equal(FaissHeaders.EnumOrdinal("faiss/impl/ScalarQuantizer.h", "RangeStat", member), (int)managed);
    }

    [Theory]
    [InlineData(IoFlags.SkipStorage, "IO_FLAG_SKIP_STORAGE")]
    [InlineData(IoFlags.ReadOnly, "IO_FLAG_READ_ONLY")]
    [InlineData(IoFlags.OnDiskSameDir, "IO_FLAG_ONDISK_SAME_DIR")]
    [InlineData(IoFlags.SkipIvfData, "IO_FLAG_SKIP_IVF_DATA")]
    [InlineData(IoFlags.SkipPrecomputeTable, "IO_FLAG_SKIP_PRECOMPUTE_TABLE")]
    [InlineData(IoFlags.PqSkipSdcTable, "IO_FLAG_PQ_SKIP_SDC_TABLE")]
    [InlineData(IoFlags.MmapIfc, "IO_FLAG_MMAP_IFC")]
    [InlineData(IoFlags.Mmap, "IO_FLAG_MMAP")]
    public void IoFlags_ValuesMatchTheFaissHeader(IoFlags managed, string constant)
    {
        Assert.Equal(FaissHeaders.IntConstant("faiss/index_io.h", constant), (int)managed);
    }

    /// <summary>
    /// The C API's own <c>FAISS_IO_FLAG_MMAP</c> is 1 (c_api/index_io_c.h), which is actually
    /// <c>IO_FLAG_SKIP_STORAGE</c>. <see cref="IoFlags"/> deliberately mirrors the C++ header instead,
    /// so a caller asking for mmap must not silently get skip-storage.
    /// </summary>
    [Fact]
    public void IoFlags_Mmap_IsNotTheMislabelledCApiMacro()
    {
        // Locals rather than constants, so the assertions read native-header value first, managed value second.
        var skipStorage = IoFlags.SkipStorage;
        var mmap = IoFlags.Mmap;
        var expected = FaissHeaders.IntConstant("faiss/index_io.h", "IO_FLAG_SKIP_IVF_DATA") | 0x646f0000;

        Assert.NotEqual(skipStorage, mmap);
        Assert.Equal(expected, (int)mmap);
    }
}
