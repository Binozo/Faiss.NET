using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Exceptions;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Indexes;

[Collection(GpuTestCollection.Name)]
public class GpuFlatIndexTests
{
    /// <summary>The device IVFFlat reconstructs batches by id, which needs no direct map, unlike the CPU index.</summary>
    [Fact]
    public async Task IvfFlat_BatchReconstruct_ReturnsStoredVectorsWithoutDirectMap()
    {
        var stored = Vectors.Random(seed: 1, count: 32, dimensions: 8);
        using var cpu = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4);
        await cpu.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8));
        cpu.Add(32, stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        Assert.Equal(stored[(4 * 8)..(20 * 8)], gpu.Reconstruct(4, 16));
        Assert.Throws<FaissNativeException>(() => gpu.Reconstruct(30, 3));
    }

    [Fact]
    public async Task IvfFlat_SingleKeyReconstructAndResidual_Throw()
    {
        using var cpu = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(8), 8, nlist: 4);
        await cpu.TrainAsync(256, Vectors.Random(seed: 98, count: 256, dimensions: 8));
        cpu.Add(8, Vectors.Random(seed: 1, count: 8, dimensions: 8));
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, 0);

        Assert.Throws<FaissNativeException>(() => gpu.Reconstruct(0));
        Assert.Throws<FaissNativeException>(() => gpu.ComputeResidual(new float[8], new float[8], 0));
    }
}
