namespace Faiss.Cpu.Search;

using System.Buffers;

/// <summary>
/// Search result that owns pooled arrays. Dispose to return arrays to pool.
/// </summary>
public ref struct SearchResultSpan : IDisposable
{
    private float[]? _distancesArray;
    private long[]? _labelsArray;
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

    public readonly int Length => _k;

    public readonly ReadOnlySpan<float> Distances
    {
        get
        {
            ThrowIfDisposed();
            return _distancesArray.AsSpan(0, _k);
        }
    }

    public readonly ReadOnlySpan<long> Labels
    {
        get
        {
            ThrowIfDisposed();
            return _labelsArray.AsSpan(0, _k);
        }
    }

    public readonly SearchResult this[int index]
    {
        get
        {
            ThrowIfDisposed();
            if ((uint)index >= (uint)_k)
                throw new ArgumentOutOfRangeException(nameof(index), index, $"Index must be < {_k}");

            return new(_labelsArray![index], _distancesArray![index]);
        }
    }

    public void Dispose()
    {
        var distances = _distancesArray;
        var labels = _labelsArray;
        _distancesArray = null;
        _labelsArray = null;

        if (distances is not null)
            ArrayPool<float>.Shared.Return(distances);
        if (labels is not null)
            ArrayPool<long>.Shared.Return(labels);
    }

    public readonly SearchResultSpan.Enumerator GetEnumerator() => new(this);

    private readonly void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(_labelsArray is null && _k != 0, typeof(SearchResultSpan));

    public ref struct Enumerator
    {
        private readonly SearchResultSpan _result;
        private int _index;

        internal Enumerator(SearchResultSpan result)
        {
            _result = result;
            _index = -1;
        }

        public readonly SearchResult Current => _result[_index];
        public bool MoveNext() => ++_index < _result.Length;
    }
}