using Faiss.Cpu.Distances;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Distances;

public class PairwiseTests
{
    private const int Dimensions = 4;
    private const int Queries = 3;
    private const int Count = 5;

    [Fact]
    public void L2Sqr_FillsRowMajorMatrixWithSquaredDistances()
    {
        var queries = Vectors.Random(seed: 70, count: Queries, dimensions: Dimensions);
        var vectors = Vectors.Random(seed: 71, count: Count, dimensions: Dimensions);
        var matrix = new float[Queries * Count];

        Pairwise.L2Sqr(Dimensions, Queries, queries, Count, vectors, matrix);

        for (var q = 0; q < Queries; q++)
        {
            for (var i = 0; i < Count; i++)
            {
                // Row-major: query q occupies matrix[q * Count .. (q + 1) * Count).
                var expected = Oracles.L2Sqr(queries.AsSpan(q * Dimensions, Dimensions), vectors.AsSpan(i * Dimensions, Dimensions));
                FloatAssert.Equal(expected, matrix[(q * Count) + i]);
            }
        }
    }

    [Fact]
    public void L2Sqr_IdenticalSets_HaveZeroDiagonal()
    {
        var vectors = Vectors.Random(seed: 72, count: Count, dimensions: Dimensions);
        var matrix = new float[Count * Count];

        Pairwise.L2Sqr(Dimensions, Count, vectors, Count, vectors, matrix);

        for (var i = 0; i < Count; i++)
        {
            FloatAssert.Equal(0f, matrix[(i * Count) + i]);

            // L2 is symmetric, so the matrix must mirror across the diagonal.
            for (var j = 0; j < Count; j++)
            {
                FloatAssert.Equal(matrix[(i * Count) + j], matrix[(j * Count) + i]);
            }
        }
    }

    [Fact]
    public void L2Sqr_InvalidArguments_Throw()
    {
        var queries = new float[Queries * Dimensions];
        var vectors = new float[Count * Dimensions];
        var matrix = new float[Queries * Count];

        Assert.Throws<ArgumentOutOfRangeException>(() => Pairwise.L2Sqr(0, Queries, queries, Count, vectors, matrix));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pairwise.L2Sqr(Dimensions, Queries + 1, queries, Count, vectors, matrix));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pairwise.L2Sqr(Dimensions, Queries, queries, Count + 1, vectors, matrix));
        Assert.Throws<ArgumentOutOfRangeException>(() => Pairwise.L2Sqr(Dimensions, Queries, queries, Count, vectors, new float[(Queries * Count) - 1]));
    }
}
