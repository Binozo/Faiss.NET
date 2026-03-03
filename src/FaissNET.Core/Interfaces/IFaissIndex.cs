namespace Faiss.Interfaces;

public interface IFaissIndex : IDisposable
{
    /// <summary>
    /// Dimensionality of the vectors in this index.
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Total number of vectors currently indexed.
    /// </summary>
    long TotalCount { get; }

    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    bool IsTrained { get; }

    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="count">The number of vectors being added.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    void Add(long count, ReadOnlySpan<float> vectors);

    /// <summary>
    /// Searches the index for the k nearest neighbors.
    /// </summary>
    /// <param name="count">The number of query vectors.</param>
    /// <param name="queryVectors">A flat span of query vectors (size: count * Dimensions).</param>
    /// <param name="k">The number of nearest neighbors to retrieve per query.</param>
    /// <param name="distances">Output span for calculated distances (size: count * k).</param>
    /// <param name="labels">Output span for vector IDs/labels (size: count * k).</param>
    void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels);

    /// <summary>
    /// Removes all elements from the index, resetting it to an empty state.
    /// </summary>
    void Reset();
}