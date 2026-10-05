using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface ITrainableFloatIndex : IFloatIndex, ITrainableIndex, INativeIndex
{
    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="count">Number of training vectors (should be at least 10x the number of centroids).</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="count"/> is negative.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="vectors"/> is too small.</exception>
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors);
}

internal static class TrainableFloatIndexImpl
{
    public static bool IsTrained(INativeIndex index) => Native.faiss_Index_is_trained(index.Handle) != 0;

    public static Task TrainAsync(INativeIndex index, long count, ReadOnlyMemory<float> vectors)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        long expected = checked(count * index.Dimensions);
        if (vectors.Length < expected)
            throw new ArgumentException($"Training span too small. Expected {expected}, got {vectors.Length}.", nameof(vectors));

        return Task.Run(() =>
        {
            unsafe
            {
                using var handle = vectors.Pin();
                float* pVectors = (float*)handle.Pointer;

                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_train(index.Handle, count, pVectors)
                );
            }
        });
    }
}