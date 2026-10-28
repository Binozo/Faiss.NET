using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

/// <summary>
/// Disposing while a search is in flight must defer the native free until the search returns.
/// Concurrent searches are supported but while a search runs it is not allowed to modify the index state (by adding vectors for instance).
/// </summary>
public class ConcurrencyContractTests
{
    private const int Threads = 4;
    private const int Iterations = 50;
    private const int Stored = 64;
    private const int Queries = 16;
    private const int K = 4;

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialFloat), MemberType = typeof(IndexCatalog))]
    public async Task ConcurrentSearches_MatchASequentialSearch(IndexCase c)
    {
        using var index = c.CreateIndex();
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Stored, Vectors.Random(seed: 600, count: Stored, dimensions: c.Dimensions));
        var queries = Vectors.Random(seed: 601, count: Queries, dimensions: c.Dimensions);

        var expectedDistances = new float[Queries * K];
        var expectedLabels = new long[Queries * K];
        index.Search(Queries, queries, K, expectedDistances, expectedLabels);

        await RunConcurrently(() =>
        {
            var distances = new float[Queries * K];
            var labels = new long[Queries * K];
            for (var i = 0; i < Iterations; i++)
            {
                Array.Clear(distances);
                Array.Clear(labels);
                index.Search(Queries, queries, K, distances, labels);

                Assert.Equal(expectedLabels, labels);
                Assert.Equal(expectedDistances, distances);
            }
        });
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.SequentialBinary), MemberType = typeof(IndexCatalog))]
    public async Task ConcurrentSearches_MatchASequentialSearch_Binary(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index).Add(Stored, Vectors.RandomBinary(seed: 602, count: Stored, bitDimensions: c.Dimensions));
        var queries = Vectors.RandomBinary(seed: 603, count: Queries, bitDimensions: c.Dimensions);

        var expectedDistances = new int[Queries * K];
        var expectedLabels = new long[Queries * K];
        index.Search(Queries, queries, K, expectedDistances, expectedLabels);

        await RunConcurrently(() =>
        {
            var distances = new int[Queries * K];
            var labels = new long[Queries * K];
            for (var i = 0; i < Iterations; i++)
            {
                Array.Clear(distances);
                Array.Clear(labels);
                index.Search(Queries, queries, K, distances, labels);

                Assert.Equal(expectedLabels, labels);
                Assert.Equal(expectedDistances, distances);
            }
        });
    }

    [Fact]
    public async Task Dispose_DuringSearch_InFlightSearchesFinishOnLiveMemory()
    {
        const int dimensions = 64;
        const int stored = 100_000;

        using var index = new IndexFlatL2(dimensions);
        index.Add(stored, Vectors.Random(seed: 604, count: stored, dimensions: dimensions));
        var queries = Vectors.Random(seed: 605, count: Queries, dimensions: dimensions);

        var expectedDistances = new float[Queries * K];
        var expectedLabels = new long[Queries * K];
        index.Search(Queries, queries, K, expectedDistances, expectedLabels);

        using var searching = new CountdownEvent(Threads);
        var disposed = 0;

        var workers = Start(() =>
        {
            var distances = new float[Queries * K];
            var labels = new long[Queries * K];
            var signalled = false;

            try
            {
                while (true)
                {
                    var afterDispose = Volatile.Read(ref disposed) == 1;
                    Array.Clear(distances);
                    Array.Clear(labels);
                    try
                    {
                        index.Search(Queries, queries, K, distances, labels);
                    }
                    catch (ObjectDisposedException)
                    {
                        return;
                    }

                    Assert.Equal(expectedLabels, labels);
                    Assert.Equal(expectedDistances, distances);

                    if (!signalled)
                    {
                        searching.Signal();
                        signalled = true;
                    }

                    if (afterDispose)
                    {
                        return;
                    }
                }
            }
            finally
            {
                // A worker that fails before its first search completes must not leave the wait below hanging.
                if (!signalled)
                {
                    searching.Signal();
                }
            }
        });

        searching.Wait(TestContext.Current.CancellationToken);
        index.Dispose();
        Volatile.Write(ref disposed, 1);

        await Task.WhenAll(workers);

        var lateDistances = new float[Queries * K];
        var lateLabels = new long[Queries * K];
        Assert.Throws<ObjectDisposedException>(() => index.Search(Queries, queries, K, lateDistances, lateLabels));
    }

    /// <summary>Runs <paramref name="body"/> on <see cref="Threads"/> dedicated threads released together.</summary>
    private static async Task RunConcurrently(Action body)
    {
        using var start = new Barrier(Threads);
        await Task.WhenAll(Start(() =>
        {
            start.SignalAndWait();
            body();
        }));
    }

    private static Task[] Start(Action body) => Enumerable.Range(0, Threads)
            .Select(_ => Task.Factory.StartNew(body, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default))
            .ToArray();
}
