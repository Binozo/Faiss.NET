using Faiss.Exceptions;
using Faiss.Interfaces;

namespace Faiss.Cpu.Interfaces;

public interface IBinaryIndex : IIndex
{
    /// <summary>
    /// Searches for the k nearest neighbors using Hamming distance.
    /// </summary>
    /// <param name="count">Number of query vectors.</param>
    /// <param name="queryVectors">Packed binary query vectors.</param>
    /// <param name="k">Number of nearest neighbors.</param>
    /// <param name="distances">Output Hamming distances (size: count * k).</param>
    /// <param name="labels">Output labels (size: count * k).</param>
    /// <exception cref="FaissException">Thrown when the search operation fails.</exception>
    void Search(long count, ReadOnlySpan<byte> queryVectors, int k, Span<int> distances, Span<long> labels);
    
    /// <summary>
    /// Assigns query vectors to nearest centroids without returning distances.
    /// </summary>
    void Assign(long count, ReadOnlySpan<byte> queryVectors, long k, Span<long> labels);
}