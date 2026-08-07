using Faiss.Exceptions;
using Faiss.Interfaces;

namespace Faiss.Cpu.Interfaces;

public interface IFloatIndex : IIndex
{
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

    /// <summary>
    /// Assigns query vectors to nearest centroids without returning distances.
    /// </summary>
    void Assign(long count, ReadOnlySpan<float> queryVectors, long k, Span<long> labels);
}