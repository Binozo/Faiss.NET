namespace Faiss.Cpu.Search;

public readonly record struct SearchResult
{
    public long Label { get; init; }
    public float Distance { get; init; }

    public SearchResult(long label, float distance)
    {
        Label = label;
        Distance = distance;
    }
}

/// <summary>
/// One search result (neighbor).
/// </summary>
public readonly record struct SearchHit(float Distance, long Label);

/// <summary>
/// Lightweight view over the results of a single query.
/// This is a ref struct, so it can contain ReadOnlySpan fields.
/// </summary>
public ref struct QueryResults
{
    public ReadOnlySpan<float> Distances;
    public ReadOnlySpan<long> Labels;
}

/// <summary>
/// High-performance search result container.
/// </summary>
public readonly struct SearchResults
{
    private readonly ReadOnlyMemory<float> _distances;
    private readonly ReadOnlyMemory<long> _labels;

    public int QueryCount { get; }
    public int K { get; }

    public ReadOnlyMemory<float> Distances => _distances;
    public ReadOnlyMemory<long> Labels => _labels;

    internal SearchResults(ReadOnlyMemory<float> distances, ReadOnlyMemory<long> labels, int queryCount, int k)
    {
        _distances = distances;
        _labels = labels;
        QueryCount = queryCount;
        K = k;
    }

    public SearchHit this[int queryIndex, int rank]
    {
        get
        {
            if ((uint)queryIndex >= (uint)QueryCount || (uint)rank >= (uint)K)
                ThrowOutOfRange();

            int idx = queryIndex * K + rank;
            return new SearchHit(_distances.Span[idx], _labels.Span[idx]);
        }
    }

    /// <summary>
    /// Returns a lightweight view (two spans) for one query.
    /// Zero allocation.
    /// </summary>
    public QueryResults GetQueryResults(int queryIndex)
    {
        if ((uint)queryIndex >= (uint)QueryCount)
            ThrowOutOfRange();

        int offset = queryIndex * K;

        return new QueryResults
        {
            Distances = _distances.Span.Slice(offset, K),
            Labels = _labels.Span.Slice(offset, K)
        };
    }

    public SearchHit[][] ToArrays()
    {
        var result = new SearchHit[QueryCount][];
        for (int q = 0; q < QueryCount; q++)
        {
            result[q] = new SearchHit[K];
            int baseIdx = q * K;
            for (int i = 0; i < K; i++)
            {
                result[q][i] = new SearchHit(
                    _distances.Span[baseIdx + i],
                    _labels.Span[baseIdx + i]);
            }
        }
        return result;
    }

    private static void ThrowOutOfRange() =>
        throw new ArgumentOutOfRangeException();
}