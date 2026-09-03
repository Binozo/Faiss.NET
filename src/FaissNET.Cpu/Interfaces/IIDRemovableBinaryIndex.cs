using Faiss.Cpu.Selectors;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IIDRemovableBinaryIndex : INativeBinaryIndex, IBinaryIndex
{
    /// <summary>
    /// Removes vectors from the index based on the provided selector.
    /// </summary>
    /// <param name="selector">The selector containing the IDs to drop.</param>
    /// <returns>The number of vectors successfully removed.</returns>
    public long RemoveIds(IDSelector selector);
}

internal static class IDRemovableBinaryIndexImpl 
{
    public static long RemoveIds(INativeBinaryIndex index, IDSelector selector)
    {
        ArgumentNullException.ThrowIfNull(selector);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_IndexBinary_remove_ids(index.Handle, selector.SafeHandle, out nuint removedCount)
        );

        return (long)removedCount;
    }
}