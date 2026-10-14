using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes;

internal static class OwnershipGuard
{
    /// <summary>
    /// Rejects an index that is a view of a part of another index, such as <c>IndexIDMap.SubIndex</c>.
    /// </summary>
    /// <remarks>
    /// That native index already belongs to its parent. Handing it over again gives it two native owners:
    /// the new composite frees it while the parent still points at it.
    /// Closing the view also strands the reference it holds on its parent, which then never gets freed.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="index"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="index"/> is a view of another index's part.</exception>
    internal static TIndex RequireIndependent<TIndex>(TIndex index, string paramName) where TIndex : INativeIndex
    {
        ArgumentNullException.ThrowIfNull(index, paramName);

        if (index.Handle is FaissBorrowedIndexHandle)
            throw new ArgumentException("This index is part of another index (for example IndexIDMap.SubIndex); pass an index you created instead.", paramName);

        return index;
    }
}
