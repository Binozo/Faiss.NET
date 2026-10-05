using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IIDSequentialBinaryIndex : INativeBinaryIndex
{
    /// <summary>
    /// Adds binary vectors to the index.
    /// </summary>
    /// <param name="count">Number of vectors being added.</param>
    /// <param name="vectors">Packed binary vectors (size: count * Dimensions / 8).</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="count"/> is negative.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="vectors"/> is too small.</exception>
    /// <exception cref="FaissUntrainedException">Thrown when the index has not been trained yet incase it requires training.</exception>
    public void Add(long count, ReadOnlySpan<byte> vectors);
}

internal static class IDSequentialBinaryIndexImpl
{
    public static unsafe void Add(INativeBinaryIndex index, long count, ReadOnlySpan<byte> vectors)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long expected = checked(count * index.CodeSize);
        if (vectors.Length < expected)
            throw new ArgumentException($"Vector span too small. Expected {expected} bytes, got {vectors.Length}.", nameof(vectors));

        fixed (byte* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinary_add(index.Handle, count, pVectors));
        }
    }
}