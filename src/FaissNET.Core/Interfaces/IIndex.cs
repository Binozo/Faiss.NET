using Faiss.Cpu.Exceptions;
using Faiss.Exceptions;
using Faiss.Models;

namespace Faiss.Interfaces;

public interface IIndex : IDisposable
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
    /// Metric type of this index.
    /// </summary>
    MetricType Metric { get; }

    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="count">The number of vectors being added.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    /// <exception cref="FaissUntrainedException">Thrown when the index has not been trained yet incase it requires training.</exception>
    /// <exception cref="FaissException">Thrown when the add operation fails.</exception>
    void Add(long count, ReadOnlySpan<float> vectors);

    /// <summary>
    /// Searches for the k nearest neighbors of the query vectors.
    /// </summary>
    /// <param name="count">The number of query vectors.</param>
    /// <param name="queryVectors">The query vectors, as a contiguous span of floats.</param>
    /// <param name="k">The number of nearest neighbors to find.</param>
    /// <param name="distances">The output distances for the nearest neighbors.</param>
    /// <param name="labels">The output labels (indices) for the nearest neighbors.</param>
    /// <exception cref="FaissException">Thrown when the search operation fails.</exception>
    void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels);

    void Assign(long count, ReadOnlySpan<float> queryVectors, long k, Span<long> labels);

    void SearchWithParams(
        long count,
        ReadOnlySpan<float> queryVectors,
        int k,
        ISearchParameters parameters,
        Span<float> distances,
        Span<long> labels);

    /// <summary>
    /// Resets the index by removing all vectors stored within it.
    /// After calling this method, the index will be empty and TotalCount will be zero.
    /// </summary>
    /// <exception cref="FaissException">Thrown when the reset operation fails.</exception>
    void Reset();
}
