namespace Faiss.Cpu.Search;

/// <summary>
/// Non-owning view of search results. Caller manages buffer lifetime.
/// </summary>
public readonly ref struct SearchResultView
{
    private readonly Span<float> _distances;
    private readonly Span<long> _labels;

    internal SearchResultView(Span<float> distances, Span<long> labels)
    {
        _distances = distances;
        _labels = labels;
    }

    public int Length => _labels.Length;

    public ReadOnlySpan<float> Distances => _distances;
    public ReadOnlySpan<long> Labels => _labels;

    public SearchResult this[int index]
    {
        get
        {
            if ((uint)index >= _labels.Length)
                throw new ArgumentOutOfRangeException(nameof(index));
            
            return new(_labels[index], _distances[index]);
        }
    }

    public Enumerator GetEnumerator() => new(this);

    public ref struct Enumerator
    {
        private readonly SearchResultView _result;
        private int _index;

        internal Enumerator(SearchResultView result)
        {
            _result = result;
            _index = -1;
        }

        public SearchResult Current => _result[_index];
        public bool MoveNext() => ++_index < _result.Length;
    }
}