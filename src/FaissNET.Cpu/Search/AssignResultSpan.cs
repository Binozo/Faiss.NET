namespace Faiss.Cpu.Search;

using System.Buffers;

public ref struct AssignResultSpan : IDisposable
{
    private long[]? _labelsArray;

    public readonly int K;
    public readonly int QueriesCount;

    public AssignResultSpan(long[] labelsArray, int k, int queriesCount = 1)
    {
        _labelsArray = labelsArray;
        K = k;
        QueriesCount = queriesCount;
    }

    public readonly ReadOnlySpan<long> Labels
    {
        get
        {
            ThrowIfDisposed();
            return _labelsArray.AsSpan(0, QueriesCount * K);
        }
    }

    public void Dispose()
    {
        var labels = _labelsArray;
        _labelsArray = null;

        if (labels is not null)
            ArrayPool<long>.Shared.Return(labels);
    }

    private readonly void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(_labelsArray is null && QueriesCount * K != 0, typeof(AssignResultSpan));
}