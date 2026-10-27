using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Refine;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Refine;

public class IndexRefineFlatTests
{
    private const int Dimensions = 8;
    private const int Count = 4;

    private static readonly float[] Data = Vectors.Random(seed: 901, count: Count, dimensions: Dimensions);

    private static void AssertFindsFirstVector(IndexRefineFlat<IndexFlatL2> index)
    {
        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, Data.AsSpan(0, Dimensions), k: 1, distances, labels);

        Assert.Equal(0, labels[0]);
        FloatAssert.Equal(0f, distances[0]);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    [InlineData(0.999f)]
    [InlineData(1000.5f)]
    public void KFactor_RejectsValuesFaissCannotReadBack(float kFactor)
    {
        using var index = new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(Dimensions));

        Assert.Throws<ArgumentOutOfRangeException>(() => index.KFactor = kFactor);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(1000f)]
    public void KFactor_AcceptsTheInclusiveBounds(float kFactor)
    {
        using var index = new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(Dimensions));

        index.KFactor = kFactor;

        Assert.Equal(kFactor, index.KFactor);
    }

    [Fact]
    public async Task RefinesApproximateCandidatesToExactDistances()
    {
        const int count = 64;

        using var index = new IndexRefineFlat<IndexPQ>(new IndexPQ(Dimensions, m: 4));
        await index.TrainAsync(512, Vectors.Random(seed: 405, count: 512, dimensions: Dimensions));
        index.KFactor = 8f;

        var vectors = Vectors.Random(seed: 406, count: count, dimensions: Dimensions);
        index.Add(count, vectors);

        var distances = new float[1];
        var labels = new long[1];

        for (var i = 0; i < count; i++)
        {
            index.Search(1, vectors.AsSpan(i * Dimensions, Dimensions), k: 1, distances, labels);

            Assert.Equal(i, labels[0]);
            FloatAssert.Equal(0f, distances[0]);
        }
    }

    [Fact]
    public void BaseIndex_Owned_IsAWorkingView()
    {
        var original = new IndexFlatL2(Dimensions);
        using var index = new IndexRefineFlat<IndexFlatL2>(original, ownBaseIndex: true);
        index.Add(Count, Data);

        Assert.Throws<ObjectDisposedException>(() => original.TotalCount);
        Assert.NotSame(original, index.BaseIndex);
        Assert.Equal(Dimensions, index.BaseIndex.Dimensions);
        Assert.Equal(Count, index.BaseIndex.TotalCount);
        Assert.True(index.OwnBaseIndex);
    }

    [Fact]
    public void BaseIndex_Owned_ThrowsOnceTheIndexIsDisposed()
    {
        var index = new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), ownBaseIndex: true);
        index.Add(Count, Data);
        var view = index.BaseIndex;

        index.Dispose();

        Assert.Throws<ObjectDisposedException>(() => view.TotalCount);
    }

    [Fact]
    public void BaseIndex_NotOwned_IsTheCallersObject()
    {
        using var original = new IndexFlatL2(Dimensions);
        using var index = new IndexRefineFlat<IndexFlatL2>(original, ownBaseIndex: false);

        Assert.Same(original, index.BaseIndex);
        Assert.False(index.OwnBaseIndex);
    }

    [Fact]
    public void NotOwned_CallerDisposesBaseEarly_IndexKeepsWorking()
    {
        var original = new IndexFlatL2(Dimensions);
        using var index = new IndexRefineFlat<IndexFlatL2>(original, ownBaseIndex: false);
        index.Add(Count, Data);

        original.Dispose();
        GC.Collect();
        GC.WaitForPendingFinalizers();

        Assert.Equal(Count, index.TotalCount);
        AssertFindsFirstVector(index);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ViewOfAnotherIndex_AsBase_IsRefused(bool ownBaseIndex)
    {
        using var map = new IndexIDMap<IndexFlatL2>(new IndexFlatL2(Dimensions), takeOwnership: true);

        Assert.Throws<ArgumentException>(() => new IndexRefineFlat<IndexFlatL2>(map.SubIndex, ownBaseIndex));

        map.Add(Count, Data, [10, 11, 12, 13]);
        Assert.Equal(Count, map.SubIndex.TotalCount);
    }
}
