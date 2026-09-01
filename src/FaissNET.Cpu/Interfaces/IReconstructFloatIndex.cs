using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface IReconstructFloatIndex : INativeIndex, IFloatIndex
{
    /// <summary>
    /// Reconstructs the original vector for a given ID.
    /// </summary>
    /// <param name="key">The ID of the vector to reconstruct.</param>
    /// <returns>The reconstructed vector.</returns>
    public float[] Reconstruct(long key);

    /// <summary>
    /// Reconstructs a batch of vectors starting from a specific ID.
    /// </summary>
    /// <param name="startKey">The starting ID of the batch.</param>
    /// <param name="count">The number of vectors to reconstruct.</param>
    /// <returns>The reconstructed vectors.</returns>
    public float[] Reconstruct(long startKey, long count);
}

internal static class ReconstructFloatIndexImpl
{
    public static float[] Reconstruct(INativeIndex index, long key)
    {
        float[] vector = new float[index.Dimensions];

        unsafe
        {
            fixed (float* pVector = vector)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_reconstruct(index.Handle, key, pVector)
                );
            }
        }

        return vector;
    }

    public static float[] Reconstruct(INativeIndex index, long startKey, long count)
    {
        float[] vectors = new float[count * index.Dimensions];

        unsafe
        {
            fixed (float* pVectors = vectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_reconstruct_n(index.Handle, startKey, count, pVectors)
                );
            }
        }

        return vectors;
    }
}