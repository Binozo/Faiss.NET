namespace Faiss.Cpu.Search;

using System.Buffers;

public struct PooledSearchResults : IDisposable
{
    private float[]? _distances;
    private long[]? _labels;
    private readonly int _k;
    private readonly int _count;

    internal PooledSearchResults(float[] distances, long[] labels, int k, int count)
    {
        _distances = distances;
        _labels = labels;
        _k = k;
        _count = count;
    }

    public readonly int Count => _count;
    public readonly int K => _k;

    public readonly SearchResult this[int queryIndex, int neighborIndex]
    {
        get
        {
            ObjectDisposedException.ThrowIf(_labels is null && _count != 0, typeof(PooledSearchResults));

            if ((uint)queryIndex >= (uint)_count)
                throw new ArgumentOutOfRangeException(
                    nameof(queryIndex), queryIndex, $"Query index must be < {_count}");

            if ((uint)neighborIndex >= (uint)_k)
                throw new ArgumentOutOfRangeException(
                    nameof(neighborIndex), neighborIndex, $"Neighbor index must be < {_k}");

            int idx = queryIndex * _k + neighborIndex;
            return new(_labels![idx], _distances![idx]);
        }
    }

    public void Dispose()
    {
        var distances = _distances;
        var labels = _labels;
        _distances = null;
        _labels = null;

        if (distances is not null)
            ArrayPool<float>.Shared.Return(distances);
        if (labels is not null)
            ArrayPool<long>.Shared.Return(labels);
    }
}