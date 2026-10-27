using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Approximate;

/// <summary>
/// The NSG index refuses fewer than 100 vectors (NNDescent.cpp:429) but the catalog only adds less than the required amount
/// thus it is tested here.
/// </summary>
public class IndexNSGTests
{
    private const int Dimensions = 8;
    private const int Count = 200;

    private static readonly float[] Data = Vectors.Random(seed: 880, count: Count, dimensions: Dimensions);

    public sealed record NSGCase(string Name, Func<IndexNSG> Create, bool Exact)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<NSGCase> Cases { get; } =
    [
        new NSGCase("NSGFlat", () => new IndexNSGFlat(Dimensions, r: 16), Exact: true),
        new NSGCase("NSGPQ", () => new IndexNSGPQ(Dimensions, r: 16, productQuantization: 4), Exact: false),
        new NSGCase("NSGSQ", () => new IndexNSGSQ(Dimensions, r: 16), Exact: false),
    ];

    public static TheoryData<NSGCase> TrainableCases { get; } = [.. Cases.Select(row => row.Data).Where(c => !c.Exact)];

    [Theory]
    [MemberData(nameof(TrainableCases))]
    public async Task Quantized_AddBeforeTraining_Throws(NSGCase c)
    {
        using var index = c.Create();
        var trainable = Assert.IsAssignableFrom<ITrainableFloatIndex>(index);

        Assert.False(trainable.IsTrained);
        Assert.Throws<FaissUntrainedException>(() => index.Add(Count, Data));

        await trainable.TrainAsync(512, Vectors.Random(seed: 881, count: 512, dimensions: Dimensions));
        Assert.True(trainable.IsTrained);
        index.Add(Count, Data);
        Assert.Equal(Count, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task EveryStoredVector_FindsItself(NSGCase c)
    {
        using var index = await Populated(c);
        var labels = new long[Count];

        index.Search(Count, Data, 1, new float[Count], labels);

        for (var i = 0; i < Count; i++)
        {
            Assert.Equal(i, labels[i]);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Reconstruct_MatchesSourceAndBatchMatchesSingleKeys(NSGCase c)
    {
        using var index = await Populated(c);
        var batch = index.Reconstruct(0, 4);

        for (var i = 0; i < 4; i++)
        {
            var single = index.Reconstruct(i);
            Assert.Equal(single, batch.AsSpan(i * Dimensions, Dimensions).ToArray());

            var source = Data.AsSpan(i * Dimensions, Dimensions);
            if (c.Exact)
            {
                FloatAssert.Equal(source, single);
            }
            else
            {
                var own = Oracles.L2Sqr(single, source);
                for (var j = 0; j < Count; j++)
                {
                    Assert.True(j == i || own < Oracles.L2Sqr(single, Data.AsSpan(j * Dimensions, Dimensions)), $"{c.Name}: vector {i} reconstructs nearer to {j}.");
                }
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task ComputeResidual_IsInputMinusReconstruction(NSGCase c)
    {
        using var index = await Populated(c);
        var query = Vectors.Random(seed: 882, count: 1, dimensions: Dimensions);
        var residual = new float[Dimensions];

        index.ComputeResidual(query, residual, key: 7);

        var stored = index.Reconstruct(7);
        for (var j = 0; j < Dimensions; j++)
        {
            FloatAssert.Equal(query[j] - stored[j], residual[j]);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Clone_KeepsTypeAndResults(NSGCase c)
    {
        using var index = await Populated(c);
        using var clone = index.Clone();

        Assert.IsType(index.GetType(), clone);

        var expected = new long[Count];
        var actual = new long[Count];
        index.Search(Count, Data, 1, new float[Count], expected);
        clone.Search(Count, Data, 1, new float[Count], actual);
        Assert.Equal(expected, actual);
    }

    private static async Task<IndexNSG> Populated(NSGCase c)
    {
        var index = c.Create();
        if (index is ITrainableFloatIndex trainable)
            await trainable.TrainAsync(512, Vectors.Random(seed: 881, count: 512, dimensions: Dimensions));

        index.Add(Count, Data);
        return index;
    }
}
