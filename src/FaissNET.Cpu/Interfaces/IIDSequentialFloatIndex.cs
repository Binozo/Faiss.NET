using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc />
public interface IIDSequentialFloatIndex : INativeIndex
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="count">The number of vectors being added.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="count"/> is negative.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="vectors"/> is too small.</exception>
    /// <exception cref="FaissUntrainedException">Thrown when the index has not been trained yet incase it requires training.</exception>
    /// <exception cref="FaissException">Thrown when the add operation fails.</exception>
    public void Add(long count, ReadOnlySpan<float> vectors);
}

internal static class IDSequentialFloatIndexImpl
{
    public static unsafe void Add(INativeIndex index, long count, ReadOnlySpan<float> vectors)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long expected = checked(count * index.Dimensions);
        if (vectors.Length < expected)
            throw new ArgumentException($"Vector span too small. Expected {expected}, got {vectors.Length}.", nameof(vectors));

        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(index.Handle, count, pVectors));
        }
    }
}