using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;

namespace Faiss.Gpu.Tests;

using Xunit;

using Cpu.Indexes;
using Gpu.Resources;

/// <summary>
/// Tests that GPU indexes can perform add and search operations correctly
/// after being transferred from CPU.
/// </summary>
[Collection("GpuSequential")]
public class GpuIndexSearchTests
{
    [Fact]
    public void AddAndSearch_SingleVector_ReturnsExactMatch()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        float[] vector = { 1.0f, 2.0f, 3.0f, 4.0f };
        gpuIndex.Add(1, vector);

        Assert.Equal(1, gpuIndex.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];
        gpuIndex.Search(1, vector, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }

    [Fact]
    public void AddAndSearch_MultipleVectors_ReturnsCorrectNearestNeighbors()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 2;
        using var cpuIndex = new IndexFlatL2(dimensions);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        float[] vectors =
        {
            0.0f, 0.0f,   // ID 0
            1.0f, 1.0f,   // ID 1
            2.0f, 2.0f,   // ID 2
            3.0f, 3.0f    // ID 3
        };
        gpuIndex.Add(4, vectors);

        Assert.Equal(4, gpuIndex.TotalCount);

        float[] query = { 1.1f, 1.1f };
        int k = 2;
        float[] distances = new float[k];
        long[] labels = new long[k];

        gpuIndex.Search(1, query, k, distances, labels);

        Assert.Equal(1, labels[0]);
        Assert.True(distances[0] < distances[1]);
    }

    [Fact]
    public void AddAndSearch_BatchQuery_ReturnsResultsForAllQueries()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 2;
        using var cpuIndex = new IndexFlatL2(dimensions);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        float[] vectors =
        {
            0.0f, 0.0f,
            1.0f, 1.0f,
            2.0f, 2.0f
        };
        gpuIndex.Add(3, vectors);

        float[] queries =
        {
            0.0f, 0.0f,
            1.1f, 1.1f
        };
        int k = 2;
        float[] distances = new float[k * 2];
        long[] labels = new long[k * 2];

        gpuIndex.Search(2, queries, k, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);
        Assert.Equal(1, labels[2]);
        Assert.Equal(2, labels[3]);
    }

    [Fact]
    public void Search_AfterReset_ReturnsZeroResults()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 2;
        using var cpuIndex = new IndexFlatL2(dimensions);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        gpuIndex.Add([1.0f, 2.0f]);
        Assert.Equal(1, gpuIndex.TotalCount);

        gpuIndex.Reset();
        Assert.Equal(0, gpuIndex.TotalCount);
    }

    [Fact]
    public void AddAndSearch_IPMetric_ReturnsCorrectInnerProduct()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 2;
        using var cpuIndex = new IndexFlatIP(dimensions);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        float[] vectors = { 2.0f, 3.0f };
        gpuIndex.Add(1, vectors);

        float[] distances = new float[1];
        long[] labels = new long[1];

        gpuIndex.Search(1, vectors, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(13.0f, distances[0]);
    }

    [Fact]
    public async Task AddAndSearch_IVFFlat_ReturnsApproximateResults()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        int nlist = 4;
        using var quantizer = new IndexFlatL2(dimensions);
        using var cpuIndex = new IndexIVFFlat(quantizer, dimensions, nlist);

        float[] trainingData = new float[200 * dimensions];
        for (int i = 0; i < trainingData.Length; i++)
        {
            trainingData[i] = (i % 100) * 0.1f;
        }

        await cpuIndex.TrainAsync(200, trainingData);

        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, cpuIndex);

        float[] target = { 10.0f, 10.0f, 10.0f, 10.0f };
        gpuIndex.Add(1, target);

        float[] distances = new float[1];
        long[] labels = new long[1];
        gpuIndex.Search(1, target, 1, distances, labels);

        Assert.Equal(0, labels[0]);
    }
}
