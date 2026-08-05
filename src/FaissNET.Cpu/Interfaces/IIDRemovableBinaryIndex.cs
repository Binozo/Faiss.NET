using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Search;

namespace Faiss.Cpu.Interfaces;

public interface IIDRemovableFloatIndex : INativeIndex, IFloatIndex
{
    /// <summary>
    /// Removes vectors from the index based on the provided selector.
    /// </summary>
    /// <param name="selector">The selector containing the IDs to drop.</param>
    /// <returns>The number of vectors successfully removed.</returns>
    public long RemoveIds(IIDSelector selector)
    {
        ArgumentNullException.ThrowIfNull(selector);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_Index_remove_ids(Handle, selector.ToNative(), out nuint removedCount)
        );

        return (long)removedCount;
    }
}