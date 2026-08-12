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
    /// Resets the index by removing all vectors stored within it.
    /// After calling this method, the index will be empty and TotalCount will be zero.
    /// </summary>
    /// <exception cref="FaissException">Thrown when the reset operation fails.</exception>
    void Reset();
}
