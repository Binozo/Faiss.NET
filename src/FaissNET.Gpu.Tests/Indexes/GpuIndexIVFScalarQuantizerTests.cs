using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Search.Parameters;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Indexes;

[Collection(GpuTestCollection.Name)]
public class GpuIndexIVFScalarQuantizerTests
{
    public static TheoryData<QuantizerType> GpuQuantizers { get; } = new(QuantizerType.QT_8bit, QuantizerType.QT_4bit, QuantizerType.QT_6bit, QuantizerType.QT_8bit_uniform, QuantizerType.QT_4bit_uniform, QuantizerType.QT_8bit_direct, QuantizerType.QT_fp16);

    /// <summary>Quantizers that keep per-dimension ranges in the scan kernel's shared memory.</summary>
    public static TheoryData<QuantizerType> PerDimensionQuantizers { get; } = new(QuantizerType.QT_8bit, QuantizerType.QT_4bit, QuantizerType.QT_6bit);

    public static TheoryData<QuantizerType> UniformQuantizers { get; } = new(QuantizerType.QT_8bit_uniform, QuantizerType.QT_4bit_uniform, QuantizerType.QT_8bit_direct, QuantizerType.QT_fp16);

    [Theory]
    [MemberData(nameof(GpuQuantizers))]
    public async Task EveryGpuQuantizer_TransfersAndMatchesCpu(QuantizerType qt)
    {
        using var cpu = await Populated(qt, dimensions: 8, count: 64);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        SearchAssert.SameResults(cpu, gpu, Data(seed: 2, count: 8, dimensions: 8), k: 5, relTol: 1e-3f, absTol: 1e-3f);
    }

    [Theory]
    [MemberData(nameof(PerDimensionQuantizers))]
    public async Task PerDimensionQuantizer_KUpTo512_Searches(QuantizerType qt)
    {
        using var cpu = await Populated(qt, dimensions: 8, count: 16);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        var (_, labels) = SearchAssert.Search(gpu, new float[8], 512);

        Assert.Equal(Enumerable.Range(0, 16).Select(i => (long)i), labels[..16].Order());
    }

    /// <summary>Above 512 the kernel's queues alone fill shared memory. Faiss would abort the process on launch.</summary>
    [Theory]
    [MemberData(nameof(PerDimensionQuantizers))]
    public async Task PerDimensionQuantizer_KAbove512_ThrowsOnEverySearchPath(QuantizerType qt)
    {
        using var cpu = await Populated(qt, dimensions: 8, count: 16);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);
        using var parameters = new SearchParametersIVF(nprobe: 4);
        var distances = new float[513];
        var labels = new long[513];

        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.Search(1, new float[8], 513, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.SearchWithParams(1, new float[8], 513, parameters, distances, labels));
        Assert.Throws<ArgumentOutOfRangeException>(() => gpu.Assign(1, new float[8], 513, labels));
    }

    /// <summary>The per-dimension ranges grow with the dimension, so wide indexes hit the limit at smaller k.</summary>
    [Theory]
    [InlineData(3000, 512, true)]
    [InlineData(3100, 512, false)]
    [InlineData(3100, 256, true)]
    public async Task PerDimensionQuantizer_SharedMemoryLimitDependsOnDimension(int dimensions, int k, bool fits)
    {
        using var cpu = await Populated(QuantizerType.QT_8bit, dimensions, count: 8);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        var search = () => SearchAssert.Search(gpu, new float[dimensions], k);

        if (fits)
            Assert.Equal(Enumerable.Range(0, 8).Select(i => (long)i), search().Labels[..8].Order());
        else
            Assert.Throws<ArgumentOutOfRangeException>(() => search());
    }

    [Theory]
    [MemberData(nameof(UniformQuantizers))]
    public async Task UniformQuantizer_SearchesUpToGpuSelectionLimit(QuantizerType qt)
    {
        using var cpu = await Populated(qt, dimensions: 8, count: 16);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        var (_, labels) = SearchAssert.Search(gpu, new float[8], 2048);

        Assert.Equal(Enumerable.Range(0, 16).Select(i => (long)i), labels[..16].Order());
    }

    /// <summary>The k limit follows the quantizer into indexes that were trained on the device.</summary>
    [Fact]
    public async Task UntrainedTransfer_KeepsTheKLimitAfterDeviceTraining()
    {
        using var cpu = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4, QuantizerType.QT_8bit);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);
        await gpu.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8));
        gpu.Add(16, Vectors.Random(seed: 1, count: 16, dimensions: 8));

        Assert.Throws<ArgumentOutOfRangeException>(() => SearchAssert.Search(gpu, new float[8], 513));
    }

    [Theory]
    [InlineData(QuantizerType.QT_bf16)]
    [InlineData(QuantizerType.QT_8bit_direct_signed)]
    public void UnsupportedQuantizer_ThrowsCloningUnsupported(QuantizerType qt)
    {
        using var cpu = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4, qt);
        using var provider = new GpuResourcesProvider();

        Assert.Throws<Faiss.Exceptions.FaissGpuCloningUnsupported>(() => GpuIndexProvider.TransferToGpu(provider, cpu, 0));
    }

    private static async Task<IndexIVFScalarQuantizer<IndexFlatL2>> Populated(QuantizerType qt, int dimensions, int count)
    {
        var cpu = new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(dimensions), dimensions, nlist: 4, qt) { Nprobe = 4 };
        await cpu.TrainAsync(256, Data(seed: 98, count: 256, dimensions: dimensions));
        cpu.Add(count, Data(seed: 1, count: count, dimensions: dimensions));
        return cpu;
    }

    /// <summary>Components spread over [0, 255], the range <see cref="QuantizerType.QT_8bit_direct"/> stores verbatim.</summary>
    private static float[] Data(int seed, int count, int dimensions) =>
        Vectors.Random(seed, count, dimensions).Select(x => 127.5f * (x + 1)).ToArray();
}
