using Faiss.Cpu.Indexes;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Infrastructure;

public static class SearchAssert
{
    public static (float[] Distances, long[] Labels) Search(FloatIndex index, float[] queries, int k)
    {
        var count = queries.Length / index.Dimensions;
        var distances = new float[count * k];
        var labels = new long[count * k];
        index.Search(count, queries, k, distances, labels);
        return (distances, labels);
    }

    /// <summary>
    /// Every label an exhaustive index holds, sorted: one query with k equal to the vector count returns each stored
    /// vector exactly once, whatever the metric.
    /// </summary>
    public static long[] StoredLabels(FloatIndex index, int count)
    {
        var (_, labels) = Search(index, new float[index.Dimensions], count);
        Assert.DoesNotContain(-1L, labels);
        return labels.Order().ToArray();
    }

    /// <summary>
    /// Every row holds the <paramref name="k"/> true nearest stored vectors, best first, with oracle distances.
    /// <paramref name="ids"/> maps a stored position to the label the index reports for it.
    /// </summary>
    public static void MatchesBruteForce(MetricType metric, float[] stored, float[] queries, int k, float[] distances, long[] labels, long[]? ids = null)
    {
        var dimensions = queries.Length / (distances.Length / k);
        var storedCount = stored.Length / dimensions;
        var queryCount = queries.Length / dimensions;

        for (var q = 0; q < queryCount; q++)
        {
            var query = queries[(q * dimensions)..((q + 1) * dimensions)];
            var expected = Enumerable.Range(0, storedCount)
                .Select(i => (Position: i, Distance: Oracle(metric, query, stored.AsSpan(i * dimensions, dimensions))))
                .OrderBy(x => metric == MetricType.InnerProduct ? -x.Distance : x.Distance)
                .Take(k)
                .ToArray();

            for (var j = 0; j < k; j++)
            {
                var row = q * k + j;
                FloatAssert.Equal(expected[j].Distance, distances[row], relTol: 1e-4f, absTol: 1e-4f);

                // Ties may legitimately swap labels, so check the label through its own distance.
                var position = ids is null ? labels[row] : Array.IndexOf(ids, labels[row]);
                Assert.InRange(position, 0, storedCount - 1);
                FloatAssert.Equal(Oracle(metric, query, stored.AsSpan((int)position * dimensions, dimensions)), distances[row], relTol: 1e-4f, absTol: 1e-4f);
            }

            Assert.Equal(k, labels.Skip(q * k).Take(k).Distinct().Count());
        }
    }

    /// <summary>
    /// Two indexes holding the same data agree: the same best neighbor per query and, rank by rank,
    /// the same distances. Labels below rank 0 may swap on near-ties, distances may not drift.
    /// </summary>
    public static void SameResults(FloatIndex expected, FloatIndex actual, float[] queries, int k, float relTol = 1e-4f, float absTol = 1e-4f)
    {
        var (expectedDistances, expectedLabels) = Search(expected, queries, k);
        var (actualDistances, actualLabels) = Search(actual, queries, k);

        for (var q = 0; q < expectedLabels.Length / k; q++)
        {
            Assert.Equal(expectedLabels[q * k], actualLabels[q * k]);
        }

        FloatAssert.Equal(expectedDistances, actualDistances, relTol, absTol);
        Assert.Equal(
            expectedLabels.Order(),
            actualLabels.Order());
    }

    public static float Oracle(MetricType metric, ReadOnlySpan<float> a, ReadOnlySpan<float> b) => metric switch
    {
        MetricType.L2 => Oracles.L2Sqr(a, b),
        MetricType.InnerProduct => Oracles.InnerProduct(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(metric), metric, null),
    };
}
