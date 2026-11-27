using System;
using System.Collections.Generic;
using System.Threading;
using Faiss.Cpu.Interfaces;
using Faiss.Interfaces;

namespace FaissNet.VectorData;

internal sealed class StoredRecord
{
    public required object Key { get; init; }
    public required Dictionary<string, object?> Data { get; init; }
}

internal abstract class FaissCollectionState : IDisposable
{
    public required string IndexKind { get; init; }

    public required string DistanceFunction { get; init; }

    public abstract IIndex Index { get; }

    public ReaderWriterLockSlim Lock { get; } = new();

    public Dictionary<object, long> KeyToId { get; } = new();
    public Dictionary<long, StoredRecord> Records { get; } = new();
    public long NextId;

    public void Dispose()
    {
        Lock.EnterWriteLock();
        try { Index.Dispose(); }
        finally { Lock.ExitWriteLock(); }
    }
}

internal class FloatFaissCollectionState(IIDMappedFloatIndex index) : FaissCollectionState
{
    public override IIDMappedFloatIndex Index { get; } = index;
}

internal class AppendOnlyFloatFaissCollectionState(IIDMappedFloatIndex index) : FloatFaissCollectionState(index)
{
    public HashSet<long> DeletedRecords { get; } = new();
}