using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc cref="ITrainableBinaryIndex"/>
public interface ITrainableBinaryIndex : IBinaryIndex, ITrainableIndex, INativeBinaryIndex
{
    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    public new bool IsTrained => Native.faiss_IndexBinary_is_trained(Handle) != 0;

    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="count">Number of training vectors (should be at least 10x the number of centroids).</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors)
    {
        return Task.Run(() =>
        {
            unsafe
            {
                using var handle = vectors.Pin();
                byte* pVectors = (byte*)handle.Pointer;

                FaissErrorHandler.ThrowIfError(
                    Native.faiss_IndexBinary_train(Handle, count, pVectors)
                );
            }
        });
    }
}