using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Selectors;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Search;

public class SearchParametersIvfTests
{
    private const int Dimensions = 8;
    private const int Nlist = 8;
    private const int Count = 64;

    [Fact]
    public void Properties_SurviveConstructionAndAssignment()
    {
        using var parameters = new SearchParametersIVF(nprobe: 5, maxCodes: 123);
        Assert.Equal(5, parameters.Nprobe);
        Assert.Equal(123, parameters.MaxCodes);

        parameters.Nprobe = int.MaxValue;
        Assert.Equal(int.MaxValue, parameters.Nprobe);

        parameters.MaxCodes = int.MaxValue;
        Assert.Equal(int.MaxValue, parameters.MaxCodes);

        parameters.MaxListsNum = 3;
        Assert.Equal(3, parameters.MaxListsNum);

        parameters.EnsureTopKFull = 1;
        Assert.Equal(1, parameters.EnsureTopKFull);

        parameters.MaxEmptyResultBuckets = 2;
        Assert.Equal(2, parameters.MaxEmptyResultBuckets);
    }

    [Fact]
    public async Task Nprobe_OverridesIndexNprobe()
    {
        const int k = 4;

        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, Nlist);
        var vectors = await PopulateAsync(index);

        index.Nprobe = 1;

        using var parameters = new SearchParametersIVF(nprobe: Nlist);
        var distances = new float[k];
        var labels = new long[k];

        index.SearchWithParams(1, vectors.AsSpan(0, Dimensions), k, parameters, distances, labels);

        Assert.Equal(BruteForce(vectors, k, allowed: null), labels);
        Assert.Equal(1, index.Nprobe);
    }

    [Fact]
    public async Task Selector_RestrictsResultsToMembers()
    {
        const int k = 3;

        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, Nlist);
        var vectors = await PopulateAsync(index);

        long[] allowed = [2, 11, 37, 58];
        using var selector = IDSelector.Batch(allowed);
        using var parameters = new SearchParametersIVF(nprobe: Nlist, maxCodes: 0, selector);

        var distances = new float[k];
        var labels = new long[k];
        index.SearchWithParams(1, vectors.AsSpan(0, Dimensions), k, parameters, distances, labels);

        Assert.Equal(BruteForce(vectors, k, allowed), labels);
    }

    [Fact]
    public async Task MaxCodes_LimitsScannedCodes()
    {
        const int k = 8;

        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, Nlist);
        var vectors = await PopulateAsync(index);

        using var unlimited = new SearchParametersIVF(nprobe: Nlist);
        using var capped = new SearchParametersIVF(nprobe: Nlist, maxCodes: 1);

        var query = vectors.AsSpan(0, Dimensions);
        var distances = new float[k];

        var unlimitedLabels = new long[k];
        index.SearchWithParams(1, query, k, unlimited, distances, unlimitedLabels);

        var cappedLabels = new long[k];
        index.SearchWithParams(1, query, k, capped, distances, cappedLabels);

        Assert.DoesNotContain(-1L, unlimitedLabels);
        Assert.True(Found(cappedLabels) < Found(unlimitedLabels), $"maxCodes:1 returned {Found(cappedLabels)} hits, unlimited returned {Found(unlimitedLabels)}.");
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var parameters = new SearchParametersIVF(nprobe: 2);
        parameters.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = parameters.Nprobe);
    }

    private static int Found(long[] labels) => labels.Count(l => l >= 0);

    private static async Task<float[]> PopulateAsync(IndexIVFFlat<IndexFlatL2> index)
    {
        var train = Vectors.Random(seed: 31, count: Count * 4, dimensions: Dimensions);
        await index.TrainAsync(Count * 4, train);

        var vectors = Vectors.Random(seed: 32, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        return vectors;
    }

    private static long[] BruteForce(float[] vectors, int k, long[]? allowed)
    {
        var query = vectors.AsSpan(0, Dimensions);
        var scored = new List<(long Id, float Score)>();

        for (long i = 0; i < Count; i++)
        {
            if (allowed is not null && !allowed.Contains(i))
            {
                continue;
            }

            scored.Add((i, Oracles.L2Sqr(query, vectors.AsSpan((int)i * Dimensions, Dimensions))));
        }

        return scored.OrderBy(s => s.Score).Take(k).Select(s => s.Id).ToArray();
    }
}
