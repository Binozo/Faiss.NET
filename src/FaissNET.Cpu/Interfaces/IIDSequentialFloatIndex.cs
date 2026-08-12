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
    /// <exception cref="FaissUntrainedException">Thrown when the index has not been trained yet incase it requires training.</exception>
    /// <exception cref="FaissException">Thrown when the add operation fails.</exception>
    public unsafe void Add(long count, ReadOnlySpan<float> vectors)
    {
        fixed (float* pVectors = vectors)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_Index_add(Handle, count, pVectors));
        }
    }
}