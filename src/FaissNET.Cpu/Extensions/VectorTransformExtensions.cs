using System.Buffers;
using Faiss.Cpu.Transforms;

namespace Faiss.Cpu.Extensions;

/// <summary>
/// Provides asynchronous extension methods for training FAISS vector transformations.
/// </summary>
public static class VectorTransformExtensions
{
    /// <summary>
    /// Asynchronously trains the vector transform using a contiguous memory block of vectors.
    /// </summary>
    /// <param name="vectorTransform">The vector transform instance to train.</param>
    /// <param name="n">The total number of vectors contained within the memory block.</param>
    /// <param name="vectors">A flat, contiguous memory block containing the vector data.</param>
    /// <returns>A task that represents the asynchronous training operation.</returns>
    public static Task TrainAsync(this VectorTransform vectorTransform, long n, ReadOnlyMemory<float> vectors)
    {
        return Task.Run(() =>
        {
            vectorTransform.Train(n, vectors.Span);
        });
    }
    
    /// <summary>
    /// Asynchronously trains the vector transform using a list of individual vectors.
    /// </summary>
    /// <remarks>
    /// This method efficiently flattens the provided list of vectors into a single, contiguous buffer 
    /// using <see cref="ArrayPool{T}.Shared"/> to minimize allocations during the training process.
    /// </remarks>
    /// <param name="vectorTransform">The vector transform instance to train.</param>
    /// <param name="vectors">A list of individual vectors, where each vector is represented as a memory block of floats.</param>
    /// <returns>A task that represents the asynchronous training operation.</returns>
    /// <exception cref="ArgumentException">Thrown when the length of any vector in the list does not match the expected input dimension (<see cref="VectorTransform.DIn"/>) of the transform.</exception>
    public static async Task TrainAsync(this VectorTransform vectorTransform, IList<ReadOnlyMemory<float>> vectors)
    {
        if (vectors.Count == 0) return;

        int dimension = vectorTransform.DIn;
        int totalFloats = vectors.Count * dimension;
        
        float[] buffer = ArrayPool<float>.Shared.Rent(totalFloats);

        try
        {
            Span<float> destination = buffer.AsSpan(0, totalFloats);

            for (int i = 0; i < vectors.Count; i++)
            {
                ReadOnlySpan<float> source = vectors[i].Span;

                if (source.Length != dimension)
                {
                    throw new ArgumentException(
                        $"Vector at index {i} has dimension {source.Length}, expected {dimension}.");
                }

                source.CopyTo(destination.Slice(i * dimension, dimension));
            }

            await vectorTransform.TrainAsync(vectors.Count, new ReadOnlyMemory<float>(buffer, 0, totalFloats));
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }
}