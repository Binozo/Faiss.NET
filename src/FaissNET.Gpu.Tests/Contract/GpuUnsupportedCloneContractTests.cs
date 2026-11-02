using Faiss.Cpu.Indexes.Approximate;
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

        Assert.Throws<FaissNativeException>(() => GpuIndexProvider.TransferToGpu(provider, cpu));

        Assert.Equal(32, cpu.TotalCount);
        Assert.Equal(32, SearchAssert.StoredLabels(cpu, 32).Length);
    }
}
