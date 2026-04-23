using Faiss.Models;

namespace Faiss.Interfaces;

public interface IBinaryIndex : IDisposable
{
    /// <summary>
    /// Dimensionality in bits. Vector byte size = Dimensions / 8.
    /// </summary>
    int Dimensions { get; }

    /// <summary>
    /// Total number of vectors currently indexed.
    /// </summary>
    long TotalCount { get; }

    /// <summary>
    /// Metric type (always Hamming for binary indexes).
    /// </summary>
    MetricType Metric { get; }

    /// <summary>
    /// Adds binary vectors to the index.
    /// </summary>
    /// <param name="count">Number of vectors being added.</param>
    /// <param name="vectors">Packed binary vectors (size: count * Dimensions / 8).</param>
    void Add(long count, ReadOnlySpan<byte> vectors);

    /// <summary>
    /// Searches for the k nearest neighbors using Hamming distance.
    /// </summary>
    /// <param name="count">Number of query vectors.</param>
    /// <param name="queryVectors">Packed binary query vectors.</param>
    /// <param name="k">Number of nearest neighbors.</param>
    /// <param name="distances">Output Hamming distances (size: count * k).</param>
    /// <param name="labels">Output labels (size: count * k).</param>
    void Search(long count, ReadOnlySpan<byte> queryVectors, int k, Span<int> distances, Span<long> labels);

    /// <summary>
    /// Assigns query vectors to nearest centroids without returning distances.
    /// </summary>
    void Assign(long count, ReadOnlySpan<byte> queryVectors, long k, Span<long> labels);

    /// <summary>
    /// Search with custom parameters (ID selector, nprobe for IVF, etc.).
    /// </summary>
    void SearchWithParams(
        long count,
        ReadOnlySpan<byte> queryVectors,
        int k,
        ISearchParameters parameters,
        Span<int> distances,
        Span<long> labels);

    /// <summary>
    /// Removes all vectors from the index.
    /// </summary>
    void Reset();
}