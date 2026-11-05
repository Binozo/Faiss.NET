using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Exceptions;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class GpuUnsupportedCloneContractTests
{
    [Theory]
    [InlineData(MetricType.L1)]
    [InlineData(MetricType.Linf)]
    public void Flat_NonL2OrIpMetric_ThrowsCloningUnsupported(MetricType metric)
    {
        using var cpu = new IndexFlat(4, metric);
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu, options));
    }

    [Fact]
    public void IvfFlat_NonL2OrIpMetric_ThrowsCloningUnsupported()
    {
        using var cpu = new IndexIVFFlat<IndexFlat>(new IndexFlat(4, MetricType.L1), 4, nlist: 2, MetricType.L1);
        using var provider = new GpuResourcesProvider();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu));
    }

    /// <summary>Only a float16 scalar quantizer has a device counterpart (a float16 GpuIndexFlat).</summary>
    [Theory]
    [InlineData(QuantizerType.QT_8bit)]
    [InlineData(QuantizerType.QT_4bit)]
    [InlineData(QuantizerType.QT_8bit_uniform)]
    [InlineData(QuantizerType.QT_bf16)]
    public void ScalarQuantizer_OtherThanFp16_ThrowsCloningUnsupported(QuantizerType qt)
    {
        using var cpu = new IndexScalarQuantizer(4, qt);
        using var provider = new GpuResourcesProvider();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu));
    }

    /// <summary>Device IVFPQ supports only 8-bit codes.</summary>
    [Fact]
    public async Task IvfPq_FourBitCodes_ThrowsAndLeavesSourceUsable()
    {
        var stored = Vectors.Random(seed: 1, count: 32, dimensions: 16);
        using var cpu = new IndexIVFPQ(16, n: 4, m: 4, subQuantizer: 4) { Nprobe = 4 };
        await cpu.TrainAsync(1024, Vectors.Random(seed: 98, count: 1024, dimensions: 16));
        cpu.Add(32, stored);
        using var provider = new GpuResourcesProvider();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu));

        Assert.Equal(32, cpu.TotalCount);
        Assert.Equal(32, SearchAssert.StoredLabels(cpu, 32).Length);
    }

    /// <summary>
    /// faiss's GPU resources are not thread safe and every shard or replica is built and searched on its own thread,
    /// so one provider cannot serve two of them.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(0, 1)]
    public void MultiGpu_SameProviderTwice_ThrowsArgumentException(int firstDevice, int secondDevice)
    {
        Assert.SkipWhen(secondDevice >= FaissGpu.GetNumGpus(), "Needs two GPUs.");
        using var cpu = new IndexFlatL2(4);
        cpu.Add(4, Vectors.Random(seed: 1, count: 4, dimensions: 4));
        using var provider = new GpuResourcesProvider();

        var sharded = Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuSharded([provider, provider], [firstDevice, secondDevice], cpu));
        var replicated = Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuReplicated([provider, provider], [firstDevice, secondDevice], cpu));

        Assert.Equal("contexts", sharded.ParamName);
        Assert.Equal("contexts", replicated.ParamName);
    }

    /// <summary>
    /// Per-dimension codecs keep two floats per dimension in 48 KiB of shared memory, so faiss refuses 6144 dimensions or more.
    /// A sharded transfer would hit that native error inside an OpenMP region and kill the process.
    /// </summary>
    [Theory]
    [InlineData(QuantizerType.QT_8bit)]
    [InlineData(QuantizerType.QT_4bit)]
    public void IvfSq_PerDimensionCodecAtSharedMemoryLimit_ThrowsCloningUnsupportedOnEveryPath(QuantizerType qt)
    {
        const int dimensions = 6144;
        using var cpu = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 2, qt, ownQuantizer: true);
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(first, cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuReplicated([first, second], [0, 0], cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu));
    }

    [Fact]
    public async Task IvfSq_PerDimensionCodecBelowSharedMemoryLimit_ShardsItsData()
    {
        const int dimensions = 6143;
        using var cpu = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 2, QuantizerType.QT_8bit, ownQuantizer: true);
        await cpu.TrainAsync(64, Vectors.Random(seed: 98, count: 64, dimensions: dimensions));
        cpu.Add(8, Vectors.Random(seed: 1, count: 8, dimensions: dimensions));
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();

        using var shards = GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu);

        Assert.Equal(dimensions, shards.Dimensions);
        Assert.Equal(8, shards.TotalCount);
        Assert.Equal([4L, 4L], [shards[0].TotalCount, shards[1].TotalCount]);
    }

    /// <summary>
    /// faiss moves the coarse quantizer to the GPU with its IVF index and only knows flat and float16 scalar quantizers.
    /// A sharded transfer would hit that native error inside an OpenMP region and kill the process.
    /// </summary>
    [Theory]
    [InlineData("HNSWFlat")]
    [InlineData("PQ")]
    [InlineData("SQ8")]
    public void Ivf_CoarseQuantizerWithoutDeviceCounterpart_ThrowsCloningUnsupportedOnEveryPath(string quantizer)
    {
        switch (quantizer)
        {
            case "HNSWFlat": AssertRejectedOnEveryPath(new IndexHNSWFlat(8)); break;
            case "PQ": AssertRejectedOnEveryPath(new IndexPQ(8, m: 4)); break;
            case "SQ8": AssertRejectedOnEveryPath(new IndexScalarQuantizer(8, QuantizerType.QT_8bit)); break;
            default: throw new ArgumentOutOfRangeException(nameof(quantizer), quantizer, null);
        }
    }

    private static void AssertRejectedOnEveryPath<T>(T quantizer) where T : class, ICpuFloatIndex, IFlatIndex, IFromNativeIndexHandle<T>
    {
        using var cpu = new IndexIVFFlat<T>(quantizer, 8, nlist: 4, ownQuantizer: true);
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();

        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(first, cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuReplicated([first, second], [0, 0], cpu));
        Assert.Throws<FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu));
        Assert.Equal(4, cpu.Nlist);
    }

    /// <summary>A float16 scalar quantizer becomes a float16 device flat index, so the IVF index transfers on every path with its data.</summary>
    [Fact]
    public async Task Ivf_CoarseQuantizerFp16_TransfersWithItsData()
    {
        const int stored = 64;
        var vectors = Vectors.Random(seed: 1, count: stored, dimensions: 8);
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: 8);
        using var cpu = new IndexIVFFlat<IndexScalarQuantizer>(new IndexScalarQuantizer(8, QuantizerType.QT_fp16), 8, nlist: 4, ownQuantizer: true) { Nprobe = 4 };
        await cpu.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8));
        cpu.Add(stored, vectors);
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();

        using var single = GpuIndexProvider.TransferToGpu(first, cpu);
        using var replicas = GpuIndexProvider.TransferToGpuReplicated([first, second], [0, 0], cpu);
        using var shards = GpuIndexProvider.TransferToGpuSharded([first, second], [0, 0], cpu);

        Assert.Equal(stored, single.TotalCount);
        Assert.Equal([stored, stored], [replicas[0].TotalCount, replicas[1].TotalCount]);
        Assert.Equal([stored / 2, stored / 2], [shards[0].TotalCount, shards[1].TotalCount]);
        SearchAssert.SameResults(cpu, single, queries, 5, relTol: 2e-3f, absTol: 1e-3f);
        SearchAssert.SameResults(cpu, replicas, queries, 5, relTol: 2e-3f, absTol: 1e-3f);
        SearchAssert.SameResults(cpu, shards, queries, 5, relTol: 2e-3f, absTol: 1e-3f);
    }
}
