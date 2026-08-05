using System.Buffers;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Extensions;

public static class IdSequentialBinaryIndexExtension
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">A flat span of vectors (size: count * Dimensions).</param>
    public static void Add(this IIdSequentialBinaryIndex index, ReadOnlySpan<byte> vectors)
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
    public static void Add(this IIdSequentialBinaryIndex index, IList<ReadOnlyMemory<byte>> vectors)
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
}