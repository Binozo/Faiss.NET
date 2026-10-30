namespace Faiss.Cpu.Interfaces;

public interface IGpuIndex : INativeIndex
{
}

public interface IGpuIndex<T> : IGpuIndex where T : INativeIndex, IFromNativeIndexHandle<T>
{
}

internal interface IGpuIVFIndex : IGpuIndex
{
    /// <summary>
    /// Set when the index was transferred with 32-bit id storage. Faiss then truncates every id added on the device to
    /// 32 bits without checking, so the wrapper has to reject wider ids itself.
    /// </summary>
    bool Uses32BitIds { get; set; }
}

internal static class GpuIVFIDs
{
    public static void ThrowIfNot32Bit(IGpuIVFIndex index, ReadOnlySpan<long> xids)
    {
        if (!index.Uses32BitIds)
            return;

        foreach (var id in xids)
        {
            if (id is < int.MinValue or > int.MaxValue)
                throw new ArgumentOutOfRangeException(nameof(xids), id, "The index stores ids in 32 bits (IndicesOptions.Bit32); this id does not fit.");
        }
    }
}
