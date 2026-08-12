using System.Buffers;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search;

namespace Faiss.Cpu.Extensions;

public static class BinaryIndexExtensions
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    public static void Add(this IIDSequentialBinaryIndex index, ReadOnlySpan<byte> vectors)
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
    public static void Add(this IIDSequentialBinaryIndex index, IList<ReadOnlyMemory<byte>> vectors)
    {
        if (vectors.Count == 0) return;

        int dimensions = index.Dimensions;
        int totalBytes = vectors.Count * dimensions;

        byte[] buffer = ArrayPool<byte>.Shared.Rent(totalBytes);

        try
        {
            Span<byte> destination = buffer.AsSpan(0, totalBytes);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<byte> source = vectors[i].Span;

                if (source.Length != dimensions)
                    throw new ArgumentException($"Vector at index {i} has dimension {source.Length}, expected {dimensions}.");

                source.CopyTo(destination.Slice(i * dimensions, dimensions));
            }

            index.Add(destination);
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Gets the nearest labels without distance.
    /// </summary>
    public static AssignResultSpan Assign(this IBinaryIndex index, ReadOnlySpan<byte> queryVectors, int k)
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
    public static void Assign(this IBinaryIndex index, ReadOnlySpan<byte> queryVector, int k, Span<long> labels)
    {
        if (queryVector.Length != index.Dimensions)
            throw new ArgumentException($"Query has {queryVector.Length} dimensions, expected {index.Dimensions}");

        if (labels.Length < k)
            throw new ArgumentException($"Labels buffer too small: {labels.Length} < {k}");

        index.Assign(1, queryVector, k, labels);
    }

    /// <summary>
    /// Trains the index using a representative sample of the dataset.
    /// </summary>
    /// <param name="index">The index to train.</param>
    /// <param name="vectors">The sample vectors to learn from.</param>
    public static async Task TrainAsync(this ITrainableBinaryIndex index, IList<ReadOnlyMemory<byte>> vectors)
    {
        if (vectors.Count == 0) return;

        int totalBytes = vectors.Count * index.Dimensions;

        byte[] buffer = ArrayPool<byte>.Shared.Rent(totalBytes);

        try
        {
            Span<byte> destination = buffer.AsSpan(0, totalBytes);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<byte> source = vectors[i].Span;

                if (source.Length != index.Dimensions)
                {
                    throw new ArgumentException(
                        $"Vector at index {i} has dimension {source.Length}, expected {index.Dimensions}.");
                }

                source.CopyTo(destination.Slice(i * index.Dimensions, index.Dimensions));
            }

            await index.TrainAsync(vectors.Count, new ReadOnlyMemory<byte>(buffer, 0, totalBytes));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}