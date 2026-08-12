using System.Buffers;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using ITrainableIndex = Faiss.Interfaces.ITrainableIndex;

namespace Faiss.Cpu.Extensions;

public static class IndexExtensions
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    public static void Add(this IIDSequentialFloatIndex index, ReadOnlySpan<float> vectors)
    {
        if (vectors.Length == 0 || vectors.Length % index.Dimensions != 0)
        {
            throw new ArgumentException($"Vector span length ({vectors.Length}) must be a multiple of dimensions ({index.Dimensions})");
        }

        index.Add(vectors.Length / index.Dimensions, vectors);
    }

    /// <summary>
    /// Adds vectors to the index from a list of memory chunks.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A list of vectors.</param>
    public static void Add(this IIDSequentialFloatIndex index, IList<ReadOnlyMemory<float>> vectors)
    {
        if (vectors.Count == 0) return;

        int dimensions = index.Dimensions;
        int totalFloats = vectors.Count * dimensions;

        float[] buffer = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            Span<float> destination = buffer.AsSpan(0, totalFloats);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != dimensions)
                    throw new ArgumentException($"Vector at index {i} has dimension {source.Length}, expected {dimensions}.");

                source.CopyTo(destination.Slice(i * dimensions, dimensions));
            }

            index.Add(destination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }
    
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">The vector to add.</param>
    /// <param name="xid">The ID to assign to the vector.</param>
    public static void Add(this IIDMappedFloatIndex index, ReadOnlySpan<float> vectors, long xid)
    {
        if (vectors.Length == 0 || vectors.Length % index.Dimensions != 0)
        {
            throw new ArgumentException($"Vector span length ({vectors.Length}) must be a multiple of dimensions ({index.Dimensions})");
        }

        index.Add(vectors.Length / index.Dimensions, vectors, new[]{xid});
    }
    
    /// <summary>
    /// Adds vectors to the index from a list of memory chunks.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A list of vectors.</param>
    /// <param name="xids">The IDs to assign to the vectors.</param>
    public static void Add(this IIDMappedFloatIndex index, IReadOnlyList<ReadOnlyMemory<float>> vectors, IReadOnlyList<long> xids)
    {
        if (vectors.Count == 0 || vectors.Count % index.Dimensions != 0)
        {
            throw new ArgumentException($"Vector span length ({vectors.Count}) must be a multiple of dimensions ({index.Dimensions})");
        }

        int dimensions = index.Dimensions;
        int totalFloats = vectors.Count * dimensions;

        float[] buffer = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            Span<float> destination = buffer.AsSpan(0, totalFloats);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != dimensions)
                    throw new ArgumentException($"Vector at index {i} has dimension {source.Length}, expected {dimensions}.");

                source.CopyTo(destination.Slice(i * dimensions, dimensions));
            }

            index.Add(vectors.Count, destination, xids.ToArray());
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Searches for the k nearest neighbors of the query vectors.
    /// </summary>
    public static SearchResultSpan Search(this IFloatIndex index, ReadOnlySpan<float> queryVectors, int k)
    {
        if (queryVectors.Length != index.Dimensions)
        {
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must exactly match dimensions ({index.Dimensions})");
        }

        long count = 1;
        var distances = ArrayPool<float>.Shared.Rent(k);
        var labels = ArrayPool<long>.Shared.Rent(k);

        try
        {
            index.Search(count, queryVectors, k, distances.AsSpan(0, k), labels.AsSpan(0, k));
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
    /// Zero-allocation search for a single query. Caller provides the stackalloc buffers.
    /// </summary>
    public static SearchResultView Search(this IFloatIndex index, ReadOnlySpan<float> queryVector, int k, Span<float> distances, Span<long> labels)
    {
        if (queryVector.Length != index.Dimensions)
            throw new ArgumentException($"Query has {queryVector.Length} dimensions, expected {index.Dimensions}");

        if (distances.Length < k)
            throw new ArgumentException($"Distances buffer too small: {distances.Length} < {k}");

        if (labels.Length < k)
            throw new ArgumentException($"Labels buffer too small: {labels.Length} < {k}");

        index.Search(1, queryVector, k, distances, labels);
        return new SearchResultView(distances, labels);
    }

    /// <summary>
    /// Searches for the k nearest neighbors of multiple query vectors at once.
    /// </summary>
    public static PooledSearchResults SearchPooled(this IFloatIndex index, ReadOnlySpan<float> queryVectors, int k)
    {
        if (queryVectors.Length == 0 || queryVectors.Length % index.Dimensions != 0)
        {
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must be a multiple of dimensions ({index.Dimensions})");
        }

        int count = queryVectors.Length / index.Dimensions;
        int resultCount = count * k;

        var distances = ArrayPool<float>.Shared.Rent(resultCount);
        var labels = ArrayPool<long>.Shared.Rent(resultCount);

        try
        {
            index.Search(count, queryVectors, k, distances.AsSpan(0, resultCount), labels.AsSpan(0, resultCount));
            return new PooledSearchResults(distances, labels, k, count);
        }
        catch
        {
            ArrayPool<float>.Shared.Return(distances);
            ArrayPool<long>.Shared.Return(labels);
            throw;
        }
    }

    public static SearchResults SearchWithParams(this IParamsFloatSearchIndex index, ReadOnlyMemory<float> queryVectors, int k, ISearchParameters parameters) => index.SearchWithParams([queryVectors], k, parameters);

    public static SearchResults SearchWithParams(this IParamsFloatSearchIndex index, IReadOnlyList<ReadOnlyMemory<float>> queryVectors, int k, ISearchParameters parameters)
    {
        int count = queryVectors.Count;

        if (count == 0 || k <= 0)
        {
            return new SearchResults(default, default, count, k);
        }

        int dim = index.Dimensions;

        for (int i = 1; i < count; i++)
        {
            if (queryVectors[i].Length != dim)
                throw new ArgumentException($"queryVectors too small. Expected {dim}, got {queryVectors[i].Length}.");
        }

        int totalQueryFloats = count * dim;

        float[] queryBuffer = ArrayPool<float>.Shared.Rent(totalQueryFloats);
        try
        {
            Span<float> flatQueries = queryBuffer.AsSpan(0, totalQueryFloats);

            int offset = 0;
            foreach (var vec in queryVectors)
            {
                vec.Span.CopyTo(flatQueries.Slice(offset, dim));
                offset += dim;
            }

            float[] distArr = new float[count * k];
            long[] labelArr = new long[count * k];

            index.SearchWithParams(count, flatQueries, k, parameters, distArr, labelArr);

            return new SearchResults(new ReadOnlyMemory<float>(distArr), new ReadOnlyMemory<long>(labelArr), count, k);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(queryBuffer);
        }
    }

    public static unsafe void RangeSearch(this IRangeSearchFloatIndex index, ReadOnlySpan<float> queryVector, float radius, RangeSearchResult result)
    {
        fixed (float* pQuery = queryVector)
        {
            FaissErrorHandler.ThrowIfError(
                Native.faiss_Index_range_search(index.Handle, 1, pQuery, radius, result.SafeHandle)
            );
        }
    }

    /// <summary>
    /// Gets the nearest labels without distance.
    /// </summary>
    public static AssignResultSpan Assign(this IFloatIndex index, ReadOnlySpan<float> queryVectors, int k)
    {
        if (queryVectors.Length == 0 || queryVectors.Length % index.Dimensions != 0)
        {
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must be a multiple of dimensions ({index.Dimensions})");
        }

        int count = queryVectors.Length / index.Dimensions;
        int resultCount = count * k;

        var labels = ArrayPool<long>.Shared.Rent(resultCount);

        try
        {
            index.Assign(count, queryVectors, k, labels.AsSpan(0, resultCount));
            return new AssignResultSpan(labels, k, count);
        }
        catch
        {
            ArrayPool<long>.Shared.Return(labels);
            throw;
        }
    }

    /// <summary>
    /// Gets the nearest labels without distance. Caller provides the stackalloc buffer.
    /// </summary>
    public static void Assign(this IFloatIndex index, ReadOnlySpan<float> queryVector, int k, Span<long> labels)
    {
        if (queryVector.Length != index.Dimensions)
            throw new ArgumentException($"Query has {queryVector.Length} dimensions, expected {index.Dimensions}");

        if (labels.Length < k)
            throw new ArgumentException($"Labels buffer too small: {labels.Length} < {k}");

        index.Assign(1, queryVector, k, labels);
    }

    /// <summary>
    /// Sets an internal index parameter via the auto-tune parameter space.
    /// </summary>
    /// <example>index.SetParameter("nprobe", 10);</example>
    private static void SetParameter(this INativeIndex index, string name, double value)
    {
        using var space = new AutoTune.ParameterSpace();
        space.SetParameter(index, name, value);
    }

    /// <summary>
    /// Trains the index using a representative sample of your dataset which can take a while.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    public static async Task TrainAsync(this ITrainableIndex index, IReadOnlyList<ReadOnlyMemory<float>> vectors)
    {
        if (vectors.Count == 0) return;

        int totalFloats = vectors.Count * index.Dimensions;

        float[] buffer = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            Span<float> destination = buffer.AsSpan(0, totalFloats);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != index.Dimensions)
                {
                    throw new ArgumentException(
                        $"Vector at index {i} has dimension {source.Length}, expected {index.Dimensions}.");
                }

                source.CopyTo(destination.Slice(i * index.Dimensions, index.Dimensions));
            }

            await index.TrainAsync(vectors.Count, new ReadOnlyMemory<float>(buffer, 0, totalFloats));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Removes vectors from the index based on the provided selector.
    /// </summary>
    /// <param name="index">The target index</param>
    /// <param name="xids">The IDs to drop.</param>
    /// <returns>The number of vectors successfully removed.</returns>
    public static long RemoveIds(this IIDRemovableFloatIndex index, ReadOnlySpan<long> xids)
    {
        using var selector = new IDSelectorBatch(xids);
        return index.RemoveIds(selector);
    }
}