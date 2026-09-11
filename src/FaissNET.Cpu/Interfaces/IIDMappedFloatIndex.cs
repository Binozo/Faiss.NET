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
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="count"/> is negative.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="vectors"/> or <paramref name="xids"/> are too small.</exception>
    public void Add(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids);
}

internal static class IDMappedFloatIndexImpl
{
    public static unsafe void Add(INativeIndex index, long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long expected = checked(count * index.Dimensions);
        if (vectors.Length < expected)
            throw new ArgumentException($"Vector span too small. Expected {expected}, got {vectors.Length}.", nameof(vectors));

        if (xids.Length < count)
        {
            throw new ArgumentException("Not enough custom IDs for the vectors.", nameof(xids));
        }

        fixed (float* pVectors = vectors)
        fixed (long* pXids = xids)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_add_with_ids(index.Handle, count, pVectors, pXids)
            );
        }
    }
}