using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class GpuDeviceIndexContractTests
{
    private const int Stored = 64;
    private const int K = 5;

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void Add_OnDevice_MatchesSameAddOnCpu(GpuIndexCase c)
    {
        var first = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var second = Vectors.Random(seed: 3, count: Stored / 2, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(first);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        ((IIDSequentialFloatIndex)gpu).Add(Stored / 2, second);
        ((IIDSequentialFloatIndex)cpu).Add(Stored / 2, second);

        Assert.Equal(Stored + Stored / 2, gpu.TotalCount);
        Assert.Equal(Enumerable.Range(0, Stored + Stored / 2).Select(i => (long)i), SearchAssert.StoredLabels(gpu, Stored + Stored / 2));
        SearchAssert.SameResults(cpu, gpu, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void AddWithIds_OnDevice_ReportsTheIds(GpuIndexCase c)
    {
        var vectors = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        var ids = Enumerable.Range(0, 8).Select(i => 9_000_000_000L + 13L * i).ToArray();
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        ((IIDMappedFloatIndex)gpu).Add(8, vectors, ids);

        Assert.Equal(ids, SearchAssert.StoredLabels(gpu, 8));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void Add_OnDevice_SurvivesTransferBackToCpu(GpuIndexCase c)
    {
        var vectors = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var ids = Enumerable.Range(0, Stored).Select(i => 500L + 3L * i).ToArray();
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        ((IIDMappedFloatIndex)gpu).Add(Stored, vectors, ids);

        using var back = c.ToCpu(gpu);

        Assert.Equal(Stored, back.TotalCount);
        Assert.Equal(0, cpu.TotalCount);
        Assert.Equal(ids, SearchAssert.StoredLabels(back, Stored));
        SearchAssert.SameResults(gpu, back, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void UntrainedTransfer_StaysUntrainedAndRejectsVectors(GpuIndexCase c)
    {
        using var cpu = c.CreateUntrainedCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);

        Assert.False(((ITrainableFloatIndex)gpu).IsTrained);
        Assert.Throws<FaissUntrainedException>(() => ((IIDSequentialFloatIndex)gpu).Add(4, vectors));
        Assert.Throws<FaissUntrainedException>(() => ((IIDMappedFloatIndex)gpu).Add(4, vectors, [1, 2, 3, 4]));
        Assert.Throws<FaissNativeException>(() => SearchAssert.Search(gpu, vectors, 1));
        Assert.Equal(0, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public async Task Train_OnDevice_ThenAddAndSearch(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        using var cpu = c.CreateUntrainedCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var trainable = (ITrainableFloatIndex)gpu;

        await trainable.TrainAsync(c.TrainCount, c.TrainingVectors());
        ((IIDSequentialFloatIndex)gpu).Add(Stored, stored);

        Assert.True(trainable.IsTrained);
        Assert.False(((ITrainableFloatIndex)cpu).IsTrained);
        Assert.Equal(Stored, gpu.TotalCount);

        // The device trained its own quantizer, so its neighbours are checked against the device index brought back.
        using var back = c.ToCpu(gpu);
        Assert.True(((ITrainableFloatIndex)back).IsTrained);
        SearchAssert.SameResults(gpu, back, Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions), K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Reset_OnDevice_EmptiesOnlyTheDeviceIndex(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        gpu.Reset();

        Assert.Equal(0, gpu.TotalCount);
        Assert.Equal(Stored, cpu.TotalCount);
        var (_, labels) = SearchAssert.Search(gpu, Vectors.Random(seed: 2, count: 2, dimensions: c.Dimensions), K);
        Assert.All(labels, label => Assert.Equal(-1, label));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceAddable), MemberType = typeof(GpuIndexCatalog))]
    public void Reset_OnDevice_KeepsTrainingForNewVectors(GpuIndexCase c)
    {
        var replacement = Vectors.Random(seed: 3, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        gpu.Reset();
        ((IIDSequentialFloatIndex)gpu).Add(8, replacement);

        Assert.True(((ITrainableFloatIndex)gpu).IsTrained);
        Assert.Equal(Enumerable.Range(0, 8).Select(i => (long)i), SearchAssert.StoredLabels(gpu, 8));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Assign_MatchesSearchLabels(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var assigned = new long[8 * K];

        gpu.Assign(8, queries, K, assigned);

        Assert.Equal(SearchAssert.Search(gpu, queries, K).Labels, assigned);
    }

    /// <summary>Faiss's GPU k-selection stops at 2048 (1024 on some ROCm targets); beyond it search fails without crashing.</summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Search_KBeyondGpuSelectionLimit_Throws(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 16, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        var error = Record.Exception(() => SearchAssert.Search(gpu, new float[c.Dimensions], 2049));

        // Per-dimension scalar quantizers are refused earlier by the binding, see GpuIndexIVFScalarQuantizerTests.
        Assert.True(error is FaissNativeException or ArgumentOutOfRangeException, $"Unexpected {error?.GetType().Name ?? "success"}.");
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Search_KFarAboveTotalCount_PadsBeyondStoredVectors(GpuIndexCase c)
    {
        // The largest k every device index type here supports; see GpuIndexIVFScalarQuantizerTests for the SQ limits.
        const int k = 512;
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 16, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        var (_, labels) = SearchAssert.Search(gpu, new float[c.Dimensions], k);

        Assert.Equal(Enumerable.Range(0, 16).Select(i => (long)i), labels[..16].Order());
        Assert.All(labels[16..], label => Assert.Equal(-1, label));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Search_LargeBatch_MatchesCpuSource(GpuIndexCase c)
    {
        const int queries = 1000;
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 2000, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        var batch = Vectors.Random(seed: 2, count: queries, dimensions: c.Dimensions);
        var (_, batchLabels) = SearchAssert.Search(gpu, batch, K);

        // Batched rows must equal the same queries searched one by one.
        for (var q = 0; q < queries; q += 97)
        {
            var (_, single) = SearchAssert.Search(gpu, batch[(q * c.Dimensions)..((q + 1) * c.Dimensions)], K);
            Assert.Equal(single, batchLabels[(q * K)..((q + 1) * K)]);
        }
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Verbose_DefaultsFalseAndRoundTrips(GpuIndexCase c)
    {
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);

        Assert.False(gpu.Verbose);
        gpu.Verbose = true;
        Assert.True(gpu.Verbose);
        Assert.False(cpu.Verbose);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceReconstructable), MemberType = typeof(GpuIndexCatalog))]
    public void Reconstruct_ReturnsStoredVectorsBitExact(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var reconstruct = (IReconstructFloatIndex)gpu;

        Assert.Equal(stored[(5 * c.Dimensions)..(6 * c.Dimensions)], reconstruct.Reconstruct(5));
        Assert.Equal(stored[(2 * c.Dimensions)..(7 * c.Dimensions)], reconstruct.Reconstruct(2, 5));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceReconstructable), MemberType = typeof(GpuIndexCatalog))]
    public void Reconstruct_OutOfRange_Throws(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var reconstruct = (IReconstructFloatIndex)gpu;

        Assert.Throws<ArgumentOutOfRangeException>(() => reconstruct.Reconstruct(-1));
        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(8));
        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(6, 3));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.DeviceReconstructable), MemberType = typeof(GpuIndexCatalog))]
    public void ComputeResidual_IsQueryMinusStoredVector(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions);
        var query = Vectors.Random(seed: 2, count: 1, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var residual = new float[c.Dimensions];

        ((IComputeResidualFloatIndex)gpu).ComputeResidual(query, residual, 3);

        FloatAssert.Equal(query.Select((x, j) => x - stored[3 * c.Dimensions + j]).ToArray(), residual);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.ParamsSearchable), MemberType = typeof(GpuIndexCatalog))]
    public void SearchWithParams_WithoutSelector_MatchesSearch(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        using var parameters = ExhaustiveParameters(c, cpu, selector: null);
        var distances = new float[8 * K];
        var labels = new long[8 * K];

        ((IParamsFloatSearchIndex)gpu).SearchWithParams(8, queries, K, parameters, distances, labels);

        var expected = SearchAssert.Search(gpu, queries, K);
        Assert.Equal(expected.Labels, labels);
        Assert.Equal(expected.Distances, distances);
    }

    /// <summary> Faiss's GPU search ignores an IDSelector and returns unfiltered neighbours. </summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.ParamsSearchable), MemberType = typeof(GpuIndexCatalog))]
    public void SearchWithParams_WithSelector_ThrowsNotSupported(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        using var selector = new IDSelectorRange(10, 20);
        using var parameters = ExhaustiveParameters(c, cpu, selector);
        var distances = new float[K];
        var labels = new long[K];

        Assert.Throws<NotSupportedException>(() => ((IParamsFloatSearchIndex)gpu).SearchWithParams(1, new float[c.Dimensions], K, parameters, distances, labels));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Ivf), MemberType = typeof(GpuIndexCatalog))]
    public void SearchWithParams_Nprobe_OverridesIndexNprobe(GpuIndexCase c)
    {
        var queries = Vectors.Random(seed: 2, count: 16, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        using var oneList = new SearchParametersIVF(nprobe: 1);

        var cpuOneList = SearchWithParams((IParamsFloatSearchIndex)cpu, queries, oneList);
        var gpuOneList = SearchWithParams((IParamsFloatSearchIndex)gpu, queries, oneList);

        Assert.NotEqual(SearchAssert.Search(cpu, queries, K).Labels, cpuOneList.Labels);
        Assert.Equal(cpuOneList.Labels.Count(l => l == -1), gpuOneList.Labels.Count(l => l == -1));
        FloatAssert.Equal(cpuOneList.Distances.Where(float.IsFinite).ToArray(), gpuOneList.Distances.Where(float.IsFinite).ToArray(),
            relTol: c.DistanceTolerance, absTol: 1e-4f);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.Ivf), MemberType = typeof(GpuIndexCatalog))]
    public void SearchWithParams_IvfGivenBaseParameters_Throws(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 8, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        using var parameters = new SearchParameters();

        Assert.Throws<FaissNativeException>(() => SearchWithParams((IParamsFloatSearchIndex)gpu, new float[c.Dimensions], parameters));
    }

    private static SearchParameters ExhaustiveParameters(GpuIndexCase c, FloatIndex cpu, IDSelector? selector) =>
        c.Has(GpuCapabilities.Ivf)
            ? new SearchParametersIVF(nprobe: ((IIVFIndex)cpu).Nlist, selector: selector)
            : new SearchParameters(selector);

    private static (float[] Distances, long[] Labels) SearchWithParams(IParamsFloatSearchIndex index, float[] queries, SearchParameters parameters)
    {
        var count = queries.Length / index.Dimensions;
        var distances = new float[count * K];
        var labels = new long[count * K];
        index.SearchWithParams(count, queries, K, parameters, distances, labels);
        return (distances, labels);
    }
}
