namespace Faiss.Cpu.Search;

using System.Buffers;

public readonly struct PooledSearchResults : IDisposable
{
    private readonly float[] _distances;
    private readonly long[] _labels;
    private readonly int _k;
    private readonly int _count;

    internal PooledSearchResults(float[] distances, long[] labels, int k, int count)
    {
        _distances = distances;
        _labels = labels;
        _k = k;
        _count = count;
    }

    public int Count => _count;
    public int K => _k;

    public SearchResult this[int queryIndex, int neighborIndex]
    {
        get
        {
            if ((uint)queryIndex >= (uint)_count)
                throw new ArgumentOutOfRangeException(
                    nameof(queryIndex), queryIndex, $"Query index must be < {_count}");

            if ((uint)neighborIndex >= (uint)_k)
                throw new ArgumentOutOfRangeException(
                    nameof(neighborIndex), neighborIndex, $"Neighbor index must be < {_k}");

            int idx = queryIndex * _k + neighborIndex;
            return new(_labels![idx], _distances[idx]);
        }
    }

    public void Dispose()
    {
        ArrayPool<float>.Shared.Return(_distances);
        ArrayPool<long>.Shared.Return(_labels);
    }
}