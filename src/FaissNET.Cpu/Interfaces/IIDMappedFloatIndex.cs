using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

/// <summary>
/// Represents an index that supports adding vectors with custom IDs.
/// </summary>
public interface IIDMappedFloatIndex : INativeIndex
{
    /// <summary>
    /// Adds vectors to the index with the specified IDs.
    /// </summary>
    /// <param name="count">The number of vectors to add.</param>
    /// <param name="vectors">The vectors to add.</param>
    /// <param name="xids">The IDs to assign to the vectors.</param>
    public unsafe void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        if (xids.Length < count)
        {
            throw new ArgumentException("Not enough custom IDs for the vectors.", nameof(xids));
        }

        fixed (float* pVectors = vectors)
        fixed (long* pXids = xids)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_add_with_ids(Handle, count, pVectors, pXids)
            );
        }
    }
}