namespace Faiss.Cpu.Search;

using System.Buffers;

/// <summary>
/// Search result that owns pooled arrays. Dispose to return arrays to pool.
/// </summary>
public readonly ref struct SearchResultSpan : IDisposable
{
    private readonly float[]? _distancesArray;
    private readonly long[]? _labelsArray;
    private readonly int _k;

    /// <summary>
    /// Creates an owning result from pooled arrays.
    /// </summary>
    internal SearchResultSpan(float[] distances, long[] labels, int k)
    {
        _distancesArray = distances;
        _labelsArray = labels;
        _k = k;
    }

    public int Length => _k;
    
    public ReadOnlySpan<float> Distances => _distancesArray.AsSpan(0, _k);
    public ReadOnlySpan<long> Labels =>  _labelsArray.AsSpan(0, _k);

    public SearchResult this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_k)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be < {_k}");

            return new(_labelsArray![index], _distancesArray![index]);
        }
    }

    public void Dispose()
    {
        var distances = _distancesArray;
        var labels = _labelsArray;

        if (distances is not null)
            ArrayPool<float>.Shared.Return(distances);
        if (labels is not null)
            ArrayPool<long>.Shared.Return(labels);
    }

    public SearchResultSpan.Enumerator GetEnumerator() => new(this);

    public ref struct Enumerator
    {
        private readonly SearchResultSpan _result;
        private int _index;

        internal Enumerator(SearchResultSpan result)
        {
            _result = result;
            _index = -1;
        }

        public SearchResult Current => _result[_index];
        public bool MoveNext() => ++_index < _result.Length;
    }
}