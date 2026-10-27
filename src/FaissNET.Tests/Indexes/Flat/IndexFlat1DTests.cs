using Faiss.Cpu.Indexes.Flat;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Flat;
public class IndexFlat1DTests
{
    [Fact]
    public void Constructor_IsAlwaysOneDimensional()
    {
        using var index = new IndexFlat1D();

        Assert.Equal(1, index.Dimensions);
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void Search_ReturnsTheNearestScalar()
    {
        using var index = new IndexFlat1D();
        float[] values = [10f, 0f, 5f, -3f];
        index.Add(4, values);

        Assert.Equal(4, index.TotalCount);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, [4.5f], 1, distances, labels);

        Assert.Equal(2, labels[0]);
    }

    [Fact]
    public void Search_DistancesAreAbsoluteDifferencesNotSquares()
    {
        using var index = new IndexFlat1D();
        index.Add(2, [0f, 10f]);

        var distances = new float[2];
        var labels = new long[2];
        index.Search(1, [0f], 2, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(1, labels[1]);

        FloatAssert.Equal(0f, distances[0]);
        FloatAssert.Equal(10f, distances[1]);
    }

    [Fact]
    public void Search_ReturnsNeighboursInAscendingDistance()
    {
        using var index = new IndexFlat1D();
        float[] values = [0f, 1f, 2f, 3f, 4f];
        index.Add(5, values);

        var distances = new float[5];
        var labels = new long[5];
        index.Search(1, [2f], 5, distances, labels);

        Assert.Equal(2, labels[0]);
        for (var i = 1; i < 5; i++)
        {
            Assert.True(distances[i] >= distances[i - 1], $"Distance {i} ({distances[i]}) precedes {distances[i - 1]}.");
        }
    }

    [Fact]
    public void ContinuousUpdate_DefaultKeepsThePermutationCurrentAcrossAdds()
    {
        using var index = new IndexFlat1D();

        index.Add(2, [5f, 1f]);
        index.Add(2, [9f, 3f]);

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, [8.5f], 1, distances, labels);

        Assert.Equal(2, labels[0]);
    }

    [Fact]
    public void UpdatePermutation_RepairsSearchWhenContinuousUpdateIsOff()
    {
        using var index = new IndexFlat1D(continuousUpdate: false);

        index.Add(4, [5f, 1f, 9f, 3f]);
        index.UpdatePermutation();

        var distances = new float[1];
        var labels = new long[1];
        index.Search(1, [8.5f], 1, distances, labels);

        Assert.Equal(2, labels[0]);
    }

    [Fact]
    public void Reconstruct_ReturnsTheStoredScalar()
    {
        using var index = new IndexFlat1D();
        float[] values = [7f, -2f];
        index.Add(2, values);

        FloatAssert.Equal([7f], index.Reconstruct(0));
        FloatAssert.Equal([-2f], index.Reconstruct(1));
    }

    [Fact]
    public void Reset_EmptiesTheIndex()
    {
        using var index = new IndexFlat1D();
        index.Add(3, [1f, 2f, 3f]);

        index.Reset();

        Assert.Equal(0, index.TotalCount);
    }
}
