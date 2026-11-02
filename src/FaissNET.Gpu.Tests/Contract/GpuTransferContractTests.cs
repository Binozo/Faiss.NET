using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

/// <summary>CPU → GPU → CPU transfers keep the data and search behaviour of the source index.</summary>
[Collection(GpuTestCollection.Name)]
public class GpuTransferContractTests
{
    private const int Stored = 64;
    private const int QueryCount = 12;
    private const int K = 5;

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_PreservesShape(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();

        using var gpu = c.ToGpu(provider, cpu, 0);

        Assert.Equal(c.Dimensions, gpu.Dimensions);
        Assert.Equal(c.Metric, gpu.Metric);
        Assert.Equal(Stored, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Exact), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_Search_MatchesBruteForce(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        var (distances, labels) = SearchAssert.Search(gpu, queries, K);

        SearchAssert.MatchesBruteForce(c.Metric, stored, queries, K, distances, labels);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.CpuDistanceParity), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_Search_MatchesCpuSource(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        SearchAssert.SameResults(cpu, gpu, Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_KExceedsTotalCount_PadsWithSentinels(GpuIndexCase c)
    {
        const int stored = 3;
        const int k = 6;
        var vectors = Vectors.Random(seed: 1, count: stored, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(vectors);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        var (_, labels) = SearchAssert.Search(gpu, vectors[..c.Dimensions], k);

        Assert.Equal([0L, 1L, 2L], labels[..stored].Order());
        Assert.All(labels[stored..], label => Assert.Equal(-1, label));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_EmptySource_SearchFindsNothing(GpuIndexCase c)
    {
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        Assert.Equal(0, gpu.TotalCount);

        var (_, labels) = SearchAssert.Search(gpu, Vectors.Random(seed: 2, count: 2, dimensions: c.Dimensions), K);

        Assert.All(labels, label => Assert.Equal(-1, label));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_LeavesSourceUnchanged(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var reference = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();

        using (c.ToGpu(provider, cpu, 0))
        {
        }

        Assert.Equal(Stored, cpu.TotalCount);
        SearchAssert.SameResults(reference, cpu, queries, K, relTol: 0, absTol: 0);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void RoundTrip_ToGpuAndBack_MatchesGpuSearch(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        using var back = c.ToCpu(gpu);

        Assert.Equal(c.Dimensions, back.Dimensions);
        Assert.Equal(c.Metric, back.Metric);
        Assert.Equal(Stored, back.TotalCount);
        SearchAssert.SameResults(gpu, back, queries, K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void RoundTrip_ResultIsIndependentOfDeviceIndex(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        var gpu = c.ToGpu(provider, cpu, 0);
        var (expectedDistances, expectedLabels) = SearchAssert.Search(gpu, queries, K);

        using var back = c.ToCpu(gpu);
        gpu.Dispose();

        var (distances, labels) = SearchAssert.Search(back, queries, K);
        Assert.Equal(expectedLabels[0], labels[0]);
        FloatAssert.Equal(expectedDistances, distances, relTol: c.DistanceTolerance, absTol: 1e-4f);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void RoundTrip_ResultAcceptsFurtherVectorsOnCpu(GpuIndexCase c)
    {
        var first = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var second = Vectors.Random(seed: 3, count: 4, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(first);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        using var back = c.ToCpu(gpu);

        ((IIDSequentialFloatIndex)back).Add(4, second);

        Assert.Equal(Stored + 4, back.TotalCount);
        var (_, labels) = SearchAssert.Search(back, second[..c.Dimensions], 1);
        Assert.Equal(Stored, labels[0]);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void TransferToGpu_Twice_GivesIndependentDeviceIndexes(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: QueryCount, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var first = c.ToGpu(provider, cpu, 0);
        using var second = c.ToGpu(provider, cpu, 0);
        var (expectedDistances, expectedLabels) = SearchAssert.Search(second, queries, K);

        first.Reset();

        Assert.Equal(0, first.TotalCount);
        Assert.Equal(Stored, second.TotalCount);
        var (distances, labels) = SearchAssert.Search(second, queries, K);
        Assert.Equal(expectedLabels, labels);
        Assert.Equal(expectedDistances, distances);
    }
}
