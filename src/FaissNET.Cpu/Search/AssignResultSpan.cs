namespace Faiss.Cpu.Search;

using System.Buffers;

public readonly ref struct AssignResultSpan : IDisposable
{
    private readonly long[] _labelsArray;

    public readonly int K;
    public readonly int QueriesCount;

    public AssignResultSpan(long[] labelsArray, int k, int queriesCount = 1)
    {
        _labelsArray = labelsArray;
        K = k;
        QueriesCount = queriesCount;
    }

    public ReadOnlySpan<long> Labels => _labelsArray.AsSpan(0, QueriesCount * K);

    public void Dispose()
    {
        if (_labelsArray != null)
        {
            ArrayPool<long>.Shared.Return(_labelsArray);
        }
    }
}