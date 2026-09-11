using System.Buffers;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search;

namespace Faiss.Cpu.Extensions;

/// <summary>
/// Convenience overloads for binary indexes. Every buffer here is measured in <em>bytes</em>
/// (<see cref="IBinaryIndex.CodeSize"/>), not in bits (<c>Dimensions</c>).
/// </summary>
public static class BinaryIndexExtensions
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A flat span of packed vectors (size: count * CodeSize).</param>
    public static void Add(this IIDSequentialBinaryIndex index, ReadOnlySpan<byte> vectors)
    {
        int codeSize = index.CodeSize;

        if (vectors.Length == 0 || vectors.Length % codeSize != 0)
        {
            throw new ArgumentException($"Vector span length ({vectors.Length}) must be a multiple of the code size ({codeSize})");
        }

        index.Add(vectors.Length / codeSize, vectors);
    }

    /// <summary>
    /// Adds vectors to the index from a list of memory chunks.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A list of packed vectors, each of CodeSize bytes.</param>
    public static void Add(this IIDSequentialBinaryIndex index, IList<ReadOnlyMemory<byte>> vectors)
    {
        if (vectors.Count == 0) return;

        int codeSize = index.CodeSize;
        int totalBytes = vectors.Count * codeSize;

        byte[] buffer = ArrayPool<byte>.Shared.Rent(totalBytes);

        try
        {
            Span<byte> destination = buffer.AsSpan(0, totalBytes);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<byte> source = vectors[i].Span;

                if (source.Length != codeSize)
                    throw new ArgumentException($"Vector at index {i} has {source.Length} bytes, expected {codeSize}.");

                source.CopyTo(destination.Slice(i * codeSize, codeSize));
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
        int codeSize = index.CodeSize;

        if (queryVectors.Length == 0 || queryVectors.Length % codeSize != 0)
        {
            throw new ArgumentException($"Vector span length ({queryVectors.Length}) must be a multiple of the code size ({codeSize})");
        }

        int count = queryVectors.Length / codeSize;
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
        if (queryVector.Length != index.CodeSize)
            throw new ArgumentException($"Query has {queryVector.Length} bytes, expected {index.CodeSize}");

        if (labels.Length < k)
            throw new ArgumentException($"Labels buffer too small: {labels.Length} < {k}");

        index.Assign(1, queryVector, k, labels);
    }

    /// <summary>
    /// Trains the index using a representative sample of the dataset.
    /// </summary>
    /// <param name="index">The index to train.</param>
    /// <param name="vectors">The sample vectors to learn from, each of CodeSize bytes.</param>
    public static async Task TrainAsync(this ITrainableBinaryIndex index, IList<ReadOnlyMemory<byte>> vectors)
    {
        if (vectors.Count == 0) return;

        int codeSize = index.CodeSize;
        int totalBytes = vectors.Count * codeSize;

        byte[] buffer = ArrayPool<byte>.Shared.Rent(totalBytes);

        try
        {
            Span<byte> destination = buffer.AsSpan(0, totalBytes);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<byte> source = vectors[i].Span;

                if (source.Length != codeSize)
                {
                    throw new ArgumentException($"Vector at index {i} has {source.Length} bytes, expected {codeSize}.");
                }

                source.CopyTo(destination.Slice(i * codeSize, codeSize));
            }

            await index.TrainAsync(vectors.Count, new ReadOnlyMemory<byte>(buffer, 0, totalBytes));
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
