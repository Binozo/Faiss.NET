using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc cref="ITrainableBinaryIndex"/>
public interface ITrainableBinaryIndex : IBinaryIndex, ITrainableIndex, INativeBinaryIndex
{
    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="count">Number of training vectors (should be at least 10x the number of centroids).</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors);
}

internal static class TrainableBinaryIndexImpl
{
    public static bool IsTrained(INativeBinaryIndex index) => Native.faiss_IndexBinary_is_trained(index.Handle) != 0;

    public static Task TrainAsync(INativeBinaryIndex index, long count, ReadOnlyMemory<byte> vectors) => Task.Run(() =>
    {
        unsafe
        {
            using var handle = vectors.Pin();
            byte* pVectors = (byte*)handle.Pointer;

            FaissErrorHandler.ThrowIfError(
                Native.faiss_IndexBinary_train(index.Handle, count, pVectors)
            );
        }
    });

}