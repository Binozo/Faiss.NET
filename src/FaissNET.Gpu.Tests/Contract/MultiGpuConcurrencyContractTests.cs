using Faiss.Cpu.Indexes;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class MultiGpuConcurrencyContractTests
{
    private const int Threads = 4;
    private const int Iterations = 50;
    private const int Stored = 64;
    private const int Queries = 16;
    private const int K = 4;

    public static TheoryData<string, MultiGpuMode> Cases { get; } = new()
    {
        { "FlatL2", MultiGpuMode.Sharded },
        { "FlatL2", MultiGpuMode.Replicated },
        { "IVFFlatL2", MultiGpuMode.Sharded },
        { "IVFFlatL2", MultiGpuMode.Replicated },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task ConcurrentSearches_MatchASequentialSearch(string name, MultiGpuMode mode)
    {
        var c = MultiGpuCatalog.Get(name);
        var queries = Vectors.Random(seed: 601, count: Queries, dimensions: c.Dimensions);
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 600, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(2);
        using var squad = c.Transfer(mode, providers, [0, 0], cpu);
        var (expectedDistances, expectedLabels) = SearchAssert.Search(squad, queries, K);

        await RunConcurrently(() =>
        {
            for (var i = 0; i < Iterations; i++)
            {
                var (distances, labels) = SearchAssert.Search(squad, queries, K);
                Assert.Equal(expectedLabels, labels);
                Assert.Equal(expectedDistances, distances);
            }
        });
    }

    [Fact]
    public async Task ConcurrentIndexerReads_ReturnOneViewPerMember()
    {
        var c = MultiGpuCatalog.Get("IVFFlatL2");
        using var cpu = c.Single.CreatePopulatedCpuIndex(Vectors.Random(seed: 600, count: Stored, dimensions: c.Dimensions));
        using var providers = new ProviderSet(3);
        using var shards = GpuIndexProvider.TransferToGpuSharded(providers.Items, [0, 0, 0], (Faiss.Cpu.Indexes.IVF.IndexIVFFlat<Faiss.Cpu.Indexes.Flat.IndexFlatL2>)cpu);
        var seen = new System.Collections.Concurrent.ConcurrentBag<FloatIndex>();

        await RunConcurrently(() =>
        {
            for (var i = 0; i < Iterations; i++)
                seen.Add(shards[i % shards.Count]);
        }, threads: 8);

        FloatIndex[] views = [shards[0], shards[1], shards[2]];
        Assert.All(seen, view => Assert.Contains(view, views, ReferenceEqualityComparer.Instance));
        Assert.Equal(3, seen.Distinct(ReferenceEqualityComparer.Instance).Count());
    }

    private static async Task RunConcurrently(Action body, int threads = Threads)
    {
        using var start = new Barrier(threads);
        await Task.WhenAll(Enumerable.Range(0, threads).Select(_ => Task.Factory.StartNew(() =>
        {
            start.SignalAndWait();
            body();
        }, TaskCreationOptions.LongRunning)));
    }
}
