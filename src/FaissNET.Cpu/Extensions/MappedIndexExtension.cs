using System.Buffers;
using Faiss.Cpu.Interfaces;

namespace Faiss.Cpu.Extensions;

public static class MappedIndexExtension
{
    /// <summary>
    /// Adds vectors to the index.
    /// </summary>
    /// <param name="index">The target index.</param>
    /// <param name="vectors">The vector to add.</param>
    /// <param name="xid">The ID to assign to the vector.</param>
    public static void Add(this IIDMappedIndex index, ReadOnlySpan<float> vectors, long xid)
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
    public static void Add(this IIDMappedIndex index, IReadOnlyList<ReadOnlyMemory<float>> vectors, IReadOnlyList<long> xids)
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
}