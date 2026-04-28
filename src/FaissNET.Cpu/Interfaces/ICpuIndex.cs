using Faiss.Interfaces;
using Faiss.Search;

namespace Faiss.Cpu.Interfaces;

/// <inheritdoc />
public interface ICpuIndex : IIndex
{
    /// <summary>
    /// Removes vectors from the index based on the provided selector.
    /// </summary>
    /// <param name="selector">The selector containing the IDs to drop.</param>
    /// <returns>The number of vectors successfully removed.</returns>
    public long RemoveIds(IDSelector selector);

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
    public float[] ReconstructBatch(long startKey, long count);
}

public interface ICpuIndex<T> : ICpuIndex where T : INativeIndex<T>, IFromNativeHandle<T>
{
    public T Clone();
}