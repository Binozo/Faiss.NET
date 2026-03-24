namespace Faiss.Cpu;

using System.Buffers;

using Search;
using Models;
using Exceptions;
using Interfaces;
using Faiss.Interfaces;

public class FaissIndex<T> : IFaissIndex where T : INativeFaissIndex
{
    protected readonly T Index;
    
    public FaissIndex(T index)
    {
        Index = index;    
    }

    /// <summary>
    /// Dimensionality of the vectors in this index.
    /// </summary>
    public int Dimensions => Index.Dimensions;
    
    /// <summary>
    /// Total number of vectors currently indexed.
    /// </summary>
    public long TotalCount => Index.TotalCount;

    /// <summary>
    /// Value indicating whether the index requires training or is already trained.
    /// </summary>
    public bool IsTrained =>  Index.IsTrained;

    /// <summary>
    /// Metric type of this index.
    /// </summary>
    public MetricType Metric => Index.Metric;
    
    public void Add(long count, ReadOnlySpan<float> vectors) 
        => Index.Add(count, vectors);

    public void Search(long count, ReadOnlySpan<float> queryVectors, int k, Span<float> distances, Span<long> labels) 
        => Index.Search(count, queryVectors, k, distances, labels);

    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    /// <exception cref="FaissException">Thrown when the add operation fails.</exception>
    public void Add(ReadOnlySpan<float> vectors)
    {
        if (vectors.Length == 0 || vectors.Length % Dimensions != 0)
        {
            // Not dividable through defined Dimensions. Corrupt data.
            throw new ArgumentException($"Vector span length ({vectors.Length}) must be a multiple of dimensions ({Dimensions})");
        }

        Index.Add(vectors.Length / Dimensions, vectors);
    }

    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="vectors">Vector to be added.</param>
    /// <exception cref="FaissException">Thrown when the add operation fails.</exception>
    public void Add(IList<ReadOnlyMemory<float>> vectors)
    {
        if (vectors.Count == 0) return;

        int totalFloats = vectors.Count * Dimensions;

        float[] buffer = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            Span<float> destination = buffer.AsSpan(0, totalFloats);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != Dimensions)
                    throw new ArgumentException(
                        $"Vector at index {i} has dimension {source.Length}, expected {Dimensions}.");

                source.CopyTo(destination.Slice(i * Dimensions, Dimensions));
            }

            Add(destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Searches for the k nearest neighbors of the query vectors.
    /// </summary>
    /// <param name="queryVectors">The query vectors.</param>
    /// <param name="k">The number of nearest neighbors to find.</param>
    /// <exception cref="FaissException">Thrown when the search operation fails.</exception>
    public SearchResultSpan Search(ReadOnlySpan<float> queryVectors, int k)
    {
        if (queryVectors.Length != Dimensions)
        {
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must be the length of dimensions ({Dimensions})");
        }

        long count = 1;
        var distances = ArrayPool<float>.Shared.Rent(k);
        var labels = ArrayPool<long>.Shared.Rent(k);
        
        try
        {
            Index.Search(count, queryVectors, k, distances.AsSpan(0, k), labels.AsSpan(0, k));
            return new SearchResultSpan(distances, labels, k);
        }
        catch
        {
            ArrayPool<float>.Shared.Return(distances);
            ArrayPool<long>.Shared.Return(labels);
            throw;
        }
    }

    /// <summary>
    /// Searches for the k nearest neighbors of the query vectors.
    /// </summary>
    /// <param name="queryVectors">The flat query vectors, as a contiguous span of floats (size: count * Dimensions).</param>
    /// <param name="k">The number of nearest neighbors to find.</param>
    /// <exception cref="FaissException">Thrown when the search operation fails.</exception>
    public PooledSearchResults SearchPooled(ReadOnlySpan<float> queryVectors, int k)
    {
        if (queryVectors.Length == 0 || queryVectors.Length % Dimensions != 0)
        {
            // Not dividable through defined Dimensions. Corrupt data.
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must be a multiple of dimensions ({Dimensions})");
        }
        
        int count =  queryVectors.Length / Dimensions;
        int resultCount = count * k;
        
        
        var distances = ArrayPool<float>.Shared.Rent(resultCount);
        var labels = ArrayPool<long>.Shared.Rent(resultCount);
        
        try
        {
            Index.Search(count, queryVectors, k, distances.AsSpan(0, resultCount), labels.AsSpan(0, resultCount));
            return new PooledSearchResults(distances, labels, k, count);
        }
        catch
        {
            ArrayPool<float>.Shared.Return(distances);
            ArrayPool<long>.Shared.Return(labels);
            throw;
        }
    }
    
    /// <summary>
    /// Zero-allocation search for single query. Caller provides buffers.
    /// Ideal for loops — allocate once, reuse many times.
    /// </summary>
    /// <example>
    /// <code>
    /// Span&lt;float&gt; dist = stackalloc float[10];
    /// Span&lt;long&gt; labels = stackalloc long[10];
    /// foreach (var query in queries)
    /// {
    ///     var result = index.Search(query, 10, dist, labels);
    /// }
    /// </code>
    /// </example>
    public SearchResultView Search(
        ReadOnlySpan<float> queryVector,
        int k,
        Span<float> distances,
        Span<long> labels)
    {
        if (queryVector.Length != Dimensions)
            throw new ArgumentException(
                $"Query has {queryVector.Length} dimensions, expected {Dimensions}");

        if (distances.Length < k)
            throw new ArgumentException($"Distances buffer too small: {distances.Length} < {k}");

        if (labels.Length < k)
            throw new ArgumentException($"Labels buffer too small: {labels.Length} < {k}");

        Index.Search(1, queryVector, k, distances, labels);
        return new SearchResultView(distances, labels);
    }

    /// <summary>
    /// Resets the index by removing all vectors stored within it.
    /// After calling this method, the index will be empty and TotalCount will be zero.
    /// </summary>
    /// <exception cref="FaissException">Thrown when the reset operation fails.</exception>
    public void Reset()
    {
        Index.Reset();
    }

    public void Dispose()
    {
        Index.Dispose();
    }
}