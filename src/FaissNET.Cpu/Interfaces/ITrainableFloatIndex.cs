using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Interfaces;

public interface ITrainableFloatIndex : IFloatIndex, ITrainableIndex, INativeIndex
{
    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    public new bool IsTrained => Native.faiss_Index_is_trained(Handle) != 0;

    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="count">Number of training vectors (should be at least 10x the number of centroids).</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors)
    {
        return Task.Run(() =>
        {
            unsafe
            {
                using var handle = vectors.Pin();
                float* pVectors = (float*)handle.Pointer;

                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_train(Handle, count, pVectors)
                );
            }
        });
    }
}