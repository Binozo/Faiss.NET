using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;

namespace Faiss.Cpu.Extensions;

public static class CpuIndexExtensions
{
    /// <summary>
    /// Removes vectors from the index based on the provided selector.
    /// </summary>
    /// <param name="cpuIndex">The target index</param>
    /// <param name="xids">The IDs to drop.</param>
    /// <returns>The number of vectors successfully removed.</returns>
    public static long RemoveIds(this ICpuIndex cpuIndex, ReadOnlySpan<long> xids)
    {
        using var selector = new IDSelectorBatch(xids);
        return cpuIndex.RemoveIds(selector);
    }
}