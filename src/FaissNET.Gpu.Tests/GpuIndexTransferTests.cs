using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;

namespace Faiss.Gpu.Tests;

using Xunit;

using Cpu.Indexes;
using Gpu.Resources;

/// <summary>
/// Tests GPU index transfer (<see cref="GpuIndexProvider.TransferToGpu{T}"/>)
/// for various CPU index types.
/// </summary>
[Collection("GpuSequential")]
public class GpuIndexTransferTests
{
    [Fact]
    public void TransferToGpu_FlatL2_SucceedsAndReturnsCorrectProperties()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);

        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        Assert.Equal(dimensions, gpuIndex.Dimensions);
        Assert.Equal(0, gpuIndex.TotalCount);
    }

    [Fact]
    public void TransferToGpu_FlatIP_SucceedsAndSearchesCorrectly()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatIP(dimensions);
        cpuIndex.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        Assert.Equal(1, gpuIndex.TotalCount);

        var results = gpuIndex.Search([1.0f, 2.0f, 3.0f, 4.0f], 1);

        Assert.Equal(0, results.Labels[0]);
    }

    [Fact]
    public void TransferToGpu_WithClonerOptions_RespectsStorageMode()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);
        cpuIndex.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        using var resources = new GpuResourcesProvider();
        using var options = new Cloning.GpuClonerOptions { StorageMode = Cloning.IndicesOptions.Bit64 };
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex, options);

        Assert.Equal(1, gpuIndex.TotalCount);
    }

    [Fact]
    public void TransferToGpu_HnswIndex_ThrowsNotImplemented()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexHNSW(dimensions, m: 16);
        cpuIndex.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        using var resources = new GpuResourcesProvider();

        var ex = Assert.Throws<Faiss.Exceptions.FaissNativeException>(() =>
            GpuIndexProvider.TransferToGpu(resources, cpuIndex));

        Assert.Contains("not implemented on GPU", ex.Message);
    }

    [Fact]
    public void TransferToCpu_RoundTrip_PreservesData()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        cpuIndex.Add(2, vectors);

        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        gpuIndex.Add(2, vectors);

        using var restored = GpuIndexProvider.TransferToCpu(gpuIndex);

        Assert.Equal(dimensions, restored.Dimensions);
        Assert.Equal(4, restored.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];
        restored.Search(1, [1.0f, 2.0f, 3.0f, 4.0f], 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }

    [Fact]
    public void TransferToGpu_NullArguments_ThrowsArgumentNullException()
    {
        using var index = new IndexFlatL2(4);
        using var resources = new GpuResourcesProvider();

        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu<IndexFlatL2>(null!, index));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu(resources, (IndexFlatL2)null!));

        using var options = new Cloning.GpuClonerOptions();
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu<IndexFlatL2>(null!, index, options));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu(resources, (IndexFlatL2)null!, options));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu(resources, index, null!));
    }

    [Fact]
    public void TransferToCpu_NullArgument_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToCpu<IndexFlatL2>(null!));
    }
}
