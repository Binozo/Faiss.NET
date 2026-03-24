namespace Faiss.Cpu;

using System.Buffers;

using Interfaces;
using Faiss.Interfaces;


public sealed class FaissIndexIDMap<T> : FaissIndex<T>, IFaissIndexWithIds where T : IFaissIndexWithIds, INativeFaissCpuIndex
{
    public FaissIndexIDMap(T index) : base(index)
    {
    }

    /// <summary>
    /// Adds vectors to the index using your own custom IDs instead of Faiss's standard sequential ones.
    /// </summary>
    /// <param name="count">The number of vectors you are adding.</param>
    /// <param name="vectors">The flat span of vector data.</param>
    /// <param name="xids">Your custom database IDs that map exactly to the vectors.</param>
    public void AddWithIds(long count, ReadOnlySpan<float> vectors, ReadOnlySpan<long> xids)
        => Index.AddWithIds(count, vectors, xids);

    /// <summary>
    /// Adds vectors to the index using your own custom IDs instead of Faiss's standard sequential ones.
    /// </summary>
    /// <param name="vectors">The flat span of vector data.</param>
    /// <param name="xids">Your custom database IDs that map exactly to the vectors.</param>
    public void AddWithIds(IList<ReadOnlyMemory<float>> vectors, IList<long> xids)
    {
        if (vectors.Count != xids.Count)
            throw new ArgumentException($"Vector count ({vectors.Count}) must match ID count ({xids.Count})");

        if (vectors.Count == 0) return;

        int totalFloats = vectors.Count * Dimensions;

        float[] vectorBuffer = ArrayPool<float>.Shared.Rent(totalFloats);
        long[] idBuffer = ArrayPool<long>.Shared.Rent(vectors.Count);

        try
        {
            Span<float> vectorDestination = vectorBuffer.AsSpan(0, totalFloats);
            Span<long> idDestination = idBuffer.AsSpan(0, vectors.Count);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != Dimensions)
                    throw new ArgumentException(
                        $"Vector at index {i} has dimension {source.Length}, expected {Dimensions}.");

                source.CopyTo(vectorDestination.Slice(i * Dimensions, Dimensions));
            }

            for (int i = 0; i < xids.Count; i++)
            {
                idDestination[i] = xids[i];
            }

            AddWithIds(vectors.Count, vectorDestination, idDestination);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(vectorBuffer);
            ArrayPool<long>.Shared.Return(idBuffer);
        }
    }
}