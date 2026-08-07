using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IReconstructBinaryIndex : INativeBinaryIndex, IBinaryIndex
{
    /// <summary>
    /// Reconstructs the original vector for a given ID.
    /// </summary>
    /// <param name="key">The ID of the vector to reconstruct.</param>
    /// <returns>The reconstructed vector.</returns>
    public byte[] Reconstruct(long key)
    {
        byte[] vector = new byte[Dimensions];

        unsafe
        {
            fixed (byte* pVector = vector)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_IndexBinary_reconstruct(Handle, key, pVector)
                );
            }
        }

        return vector;
    }

    /// <summary>
    /// Reconstructs a batch of vectors starting from a specific ID.
    /// </summary>
    /// <param name="startKey">The starting ID of the batch.</param>
    /// <param name="count">The number of vectors to reconstruct.</param>
    /// <returns>The reconstructed vectors.</returns>
    public byte[] Reconstruct(long startKey, long count)
    {
        byte[] vectors = new byte[count * Dimensions];

        unsafe
        {
            fixed (byte* pVectors = vectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_IndexBinary_reconstruct_n(Handle, startKey, count, pVectors)
                );
            }
        }

        return vectors;
    }
}