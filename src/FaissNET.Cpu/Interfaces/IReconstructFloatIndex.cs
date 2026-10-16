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
        ArgumentOutOfRangeException.ThrowIfNegative(key);

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
        ArgumentOutOfRangeException.ThrowIfNegative(startKey);
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        KeyLayout layout = KeyLayout.Of(index);
        if (layout is { NativeChecksBatchReconstruct: false, KeysAreIds: false })
            ThrowIfRangeExceedsCount(index, startKey, count);

        float[] vectors = new float[checked(count * index.Dimensions)];

        unsafe
        {
            fixed (float* pVectors = vectors)
            {
                if (layout.NativeChecksBatchReconstruct || !layout.KeysAreIds)
                {
                    FaissErrorHandler.ThrowIfError(Native.faiss_Index_reconstruct_n(index.Handle, startKey, count, pVectors));
                }
                else
                {
                    // As the id existence cannot be checked up front this batch call must be splitted into individual calls
                    // because this way they fail cleanly instead of causing a process kill caused by the OpenMP region
                    int dimensions = index.Dimensions;
                    for (long i = 0; i < count; i++)
                    {
                        FaissErrorHandler.ThrowIfError(Native.faiss_Index_reconstruct(index.Handle, startKey + i, pVectors + i * dimensions));
                    }
                }
            }
        }

        return vectors;
    }

    /// <summary>Rejects <c>[startKey, startKey + count)</c> unless every position in it holds a vector.</summary>
    internal static void ThrowIfRangeExceedsCount(INativeIndex index, long startKey, long count)
    {
        long total = index.TotalCount;
        if (count > total - startKey)
            throw new ArgumentOutOfRangeException(nameof(count), count, $"Range [{startKey}, {startKey} + {count}) exceeds the {total} vectors in the index.");
    }
}
