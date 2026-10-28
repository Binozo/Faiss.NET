using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class RangeSearchContractTests
{
    private const int Count = 8;

    [Theory]
    [MemberData(nameof(IndexCatalog.RangeSearchFloat), MemberType = typeof(IndexCatalog))]
    public void RangeSearch_ReturnsExactlyTheVectorsInsideRadiusWithTheirDistances(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Populate(index, c);

        var radius = Median(c, vectors);

        using var result = new RangeSearchResult(Count);
        Range(index).RangeSearch(Count, vectors, radius, result);

        Assert.Equal(Count, result.Nq);

        for (var q = 0; q < Count; q++)
        {
            var query = vectors.AsSpan(q * c.Dimensions, c.Dimensions);
            var expected = Inside(c, query, vectors, radius);
            var hit = result.GetQueryResult(q);

            Assert.Equal(expected, hit.Labels.ToArray().Order().ToArray());
            Assert.Equal(expected.Length, hit.Distances.Length);

            for (var i = 0; i < hit.Labels.Length; i++)
            {
                var neighbour = vectors.AsSpan((int)hit.Labels[i] * c.Dimensions, c.Dimensions);
                FloatAssert.Equal(Oracle(c.Metric, query, neighbour), hit.Distances[i], relTol: 1e-4f, absTol: 1e-5f);
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.RangeSearchFloat), MemberType = typeof(IndexCatalog))]
    public void RangeSearch_EmptyIndex_ReturnsNoHits(IndexCase c)
    {
        using var index = c.CreateIndex();
        var query = Vectors.Random(seed: 21, count: 1, dimensions: c.Dimensions);

        using var result = new RangeSearchResult(1);
        Range(index).RangeSearch(1, query, radius: 1e9f, result);

        Assert.Equal(1, result.Nq);
        Assert.Empty(result.GetQueryResult(0).Labels.ToArray());
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ApproximateRangeSearchFloat), MemberType = typeof(IndexCatalog))]
    public void RangeSearch_Approximate_ReturnsOnlyRealHitsInsideTheRadius(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var radius = Median(c, vectors);

        using var result = new RangeSearchResult(Count);
        Range(index).RangeSearch(Count, vectors, radius, result);

        Assert.Equal(Count, result.Nq);

        var total = 0;

        for (var q = 0; q < Count; q++)
        {
            var query = vectors.AsSpan(q * c.Dimensions, c.Dimensions);
            var truth = Inside(c, query, vectors, radius);
            var hit = result.GetQueryResult(q);

            Assert.Equal(hit.Labels.Length, hit.Distances.Length);
            Assert.Equal(hit.Labels.Length, hit.Labels.ToArray().Distinct().Count());

            for (var i = 0; i < hit.Labels.Length; i++)
            {
                var label = hit.Labels[i];

                Assert.InRange(label, 0L, Count - 1L);
                Assert.Contains(label, truth);

                var neighbour = vectors.AsSpan((int)label * c.Dimensions, c.Dimensions);
                FloatAssert.Equal(Oracle(c.Metric, query, neighbour), hit.Distances[i], relTol: 1e-4f, absTol: 1e-5f);
            }

            total += hit.Labels.Length;
        }

        Assert.True(total > 0, "Approximate range search returned no hits for any query; the radius or dataset is wrong.");
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.LossyRangeSearchFloat), MemberType = typeof(IndexCatalog))]
    public void RangeSearch_Lossy_ReturnsOnlyHitsItScoredInsideTheRadius(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var radius = Median(c, vectors);

        using var result = new RangeSearchResult(Count);
        Range(index).RangeSearch(Count, vectors, radius, result);

        Assert.Equal(Count, result.Nq);

        var total = 0;

        for (var q = 0; q < Count; q++)
        {
            var hit = result.GetQueryResult(q);

            Assert.Equal(hit.Labels.Length, hit.Distances.Length);
            Assert.Equal(hit.Labels.Length, hit.Labels.ToArray().Distinct().Count());

            for (var i = 0; i < hit.Labels.Length; i++)
            {
                Assert.InRange(hit.Labels[i], 0L, Count - 1L);
                Assert.True(hit.Distances[i] < radius, $"Hit {hit.Labels[i]} scored {hit.Distances[i]} outside radius {radius}.");
            }

            total += hit.Labels.Length;
        }

        Assert.True(total > 0, "Quantized range search returned no hits at all; the radius or dataset is wrong.");
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(4f)]
    [InlineData(8f)]
    [InlineData(16f)]
    [InlineData(33f)]
    [InlineData(1e9f)]
    public void RangeSearch_Lsh_ComparesSquaredL2AgainstDecodedVectors(float radius)
    {
        const int dimensions = 8;

        using var index = new IndexLSH(dimensions, nbits: 32);
        var vectors = Vectors.Random(seed: 20, count: Count, dimensions: dimensions);
        index.Add(Count, vectors);

        var decoded = index.Reconstruct(0, Count);

        using var result = new RangeSearchResult(Count);
        index.RangeSearch(Count, vectors, radius, result);

        for (var q = 0; q < Count; q++)
        {
            var query = vectors.AsSpan(q * dimensions, dimensions);

            var expected = new List<long>();
            for (var i = 0; i < Count; i++)
            {
                if (Oracles.L2Sqr(query, decoded.AsSpan(i * dimensions, dimensions)) < radius)
                {
                    expected.Add(i);
                }
            }

            var hit = result.GetQueryResult(q);

            Assert.Equal(expected, hit.Labels.ToArray().Order().ToArray());

            for (var i = 0; i < hit.Labels.Length; i++)
            {
                var neighbour = decoded.AsSpan((int)hit.Labels[i] * dimensions, dimensions);
                FloatAssert.Equal(Oracles.L2Sqr(query, neighbour), hit.Distances[i], relTol: 1e-4f, absTol: 1e-5f);
            }
        }
    }

    [Fact]
    public void RangeSearch_Binary_ReturnsVectorsInsideHammingRadius()
    {
        const int bits = 16;

        using var index = new IndexBinaryFlat(bits);
        var vectors = Vectors.RandomBinary(seed: 22, count: Count, bitDimensions: bits);
        index.Add(Count, vectors);

        const byte radius = 6;
        var codeSize = bits / 8;

        using var result = new RangeSearchResult(Count);
        index.RangeSearch(Count, vectors, radius, result);

        for (var q = 0; q < Count; q++)
        {
            var query = vectors.AsSpan(q * codeSize, codeSize);

            var expected = new List<long>();
            for (var i = 0; i < Count; i++)
            {
                if (Oracles.Hamming(query, vectors.AsSpan(i * codeSize, codeSize)) < radius)
                {
                    expected.Add(i);
                }
            }

            var actual = result.GetQueryResult(q);

            Assert.Equal(expected, actual.Labels.ToArray().Order().ToArray());
        }
    }

    [Fact]
    public void GetQueryResult_OutOfRange_Throws()
    {
        using var result = new RangeSearchResult(2);

        Assert.Equal(2, result.Nq);
        Assert.Throws<ArgumentOutOfRangeException>(() => result.GetQueryResult(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => result.GetQueryResult(2));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.RangeSearchFloat), MemberType = typeof(IndexCatalog))]
    public void InvalidArguments_Throw(IndexCase c)
    {
        const int queries = 2;

        using var index = c.CreateIndex();
        var vectors = Populate(index, c);
        var range = Range(index);

        using var result = new RangeSearchResult(queries);

        Assert.Throws<ArgumentNullException>(() => range.RangeSearch(queries, vectors, 1f, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => range.RangeSearch(-1, vectors, 1f, result));
        Assert.Throws<ArgumentException>(() => range.RangeSearch(queries, vectors.AsSpan(0, c.Dimensions), 1f, result));

        // Result sized for fewer queries than requested
        using var tooSmall = new RangeSearchResult(1);
        Assert.Throws<ArgumentException>(() => range.RangeSearch(queries, vectors, 1f, tooSmall));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.Binary), MemberType = typeof(IndexCatalog))]
    public void InvalidArguments_Binary_Throw(BinaryIndexCase c)
    {
        const int queries = 2;

        using var index = c.CreateIndex();
        var range = Assert.IsAssignableFrom<IRangeSearchBinaryIndex>(index);
        var vectors = Vectors.RandomBinary(seed: 23, count: queries, bitDimensions: c.Dimensions);

        using var result = new RangeSearchResult(queries);

        Assert.Throws<ArgumentNullException>(() => range.RangeSearch(queries, vectors, 1, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => range.RangeSearch(-1, vectors, 1, result));
        Assert.Throws<ArgumentException>(() => range.RangeSearch(queries, vectors.AsSpan(0, c.CodeSize), 1, result));

        // Result sized for fewer queries than requested
        using var tooSmall = new RangeSearchResult(1);
        Assert.Throws<ArgumentException>(() => range.RangeSearch(queries, vectors, 1, tooSmall));
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var result = new RangeSearchResult(1);
        result.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = result.Nq);
    }

    private static float[] Populate(FloatIndex index, IndexCase c)
    {
        var vectors = Vectors.Random(seed: 20, count: Count, dimensions: c.Dimensions);
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Count, vectors);
        return vectors;
    }

    private static IRangeSearchFloatIndex Range(FloatIndex index) => Assert.IsAssignableFrom<IRangeSearchFloatIndex>(index);

    private static float Oracle(MetricType metric, ReadOnlySpan<float> a, ReadOnlySpan<float> b) => metric == MetricType.InnerProduct ? Oracles.InnerProduct(a, b) : Oracles.L2Sqr(a, b);

    /// <summary>Ids whose distance to <paramref name="query"/> falls inside <paramref name="radius"/>, ascending.</summary>
    private static long[] Inside(IndexCase c, ReadOnlySpan<float> query, float[] vectors, float radius)
    {
        var inside = new List<long>();
        for (var i = 0; i < Count; i++)
        {
            var score = Oracle(c.Metric, query, vectors.AsSpan(i * c.Dimensions, c.Dimensions));

            // L2 keeps what is nearer than the radius; inner product keeps what scores higher.
            if (c.Metric == MetricType.InnerProduct ? score > radius : score < radius)
            {
                inside.Add(i);
            }
        }

        return [.. inside];
    }

    /// <summary>
    /// A radius that splits the dataset without landing on a stored distance.
    /// </summary>
    private static float Median(IndexCase c, float[] vectors)
    {
        var scores = new float[Count];
        for (var i = 0; i < Count; i++)
        {
            scores[i] = Oracle(c.Metric, vectors.AsSpan(0, c.Dimensions), vectors.AsSpan(i * c.Dimensions, c.Dimensions));
        }

        Array.Sort(scores);

        var low = scores[(Count / 2) - 1];
        var high = scores[Count / 2];
        Assert.True(high - low > 1e-3f, "Dataset has near-tied distances; pick another seed.");

        return low + ((high - low) / 2f);
    }
}
