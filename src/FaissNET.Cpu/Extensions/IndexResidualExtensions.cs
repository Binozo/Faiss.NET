using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Extensions;

using System;
using Interfaces;

/// <summary>
/// Provides extension methods for calculating the residual error between 
/// original uncompressed vectors and their reconstructed counterparts within a FAISS index.
/// </summary>
public static class IndexResidualExtensions
{
    /// <summary>
    /// Computes the residual error vector for a single input vector relative to its reconstructed index entry.
    /// </summary>
    /// <remarks>
    /// The residual is calculated as the difference between the original vector and the 
    /// reconstructed vector associated with the specified key: residual = x - reconstruct(key).
    /// </remarks>
    /// <param name="index">The native FAISS index instance.</param>
    /// <param name="originalVector">A span containing the original, uncompressed vector data.</param>
    /// <param name="residualVector">A span where the computed residual vector will be written.</param>
    /// <param name="key">The key (ID) of the vector in the index to reconstruct and compare against.</param>
    /// <exception cref="ArgumentException">Thrown when the length of <paramref name="originalVector"/> or <paramref name="residualVector"/> does not match the dimensions of the index.</exception>
    public static void ComputeResidual(
        this INativeIndex index,
        ReadOnlySpan<float> originalVector,
        Span<float> residualVector,
        long key)
    {
        if (originalVector.Length != index.Dimensions || residualVector.Length != index.Dimensions)
        {
            throw new ArgumentException(
                $"Vector lengths must match index dimensions ({index.Dimensions}).");
        }

        unsafe
        {
            fixed (float* pX = originalVector)
            fixed (float* pRes = residualVector)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_compute_residual(
                        index.Handle, pX, pRes, key)
                );
            }
        }
    }

    /// <summary>
    /// Computes the residual error vectors for a batch of input vectors relative to their reconstructed index entries.
    /// </summary>
    /// <param name="index">The native FAISS index instance.</param>
    /// <param name="originalVectors">A contiguous span containing the batch of original, uncompressed vectors.</param>
    /// <param name="residualVectors">A contiguous span where the computed batch of residual vectors will be written.</param>
    /// <param name="keys">A span of keys (IDs) corresponding to the vectors in the index.</param>
    /// <exception cref="ArgumentException">Thrown when the length of <paramref name="originalVectors"/> or <paramref name="residualVectors"/> is insufficient for the batch size and index dimensions.</exception>
    public static void ComputeResidualBatch(
        this INativeIndex index,
        ReadOnlySpan<float> originalVectors,
        Span<float> residualVectors,
        ReadOnlySpan<long> keys)
    {
        long count = keys.Length;
        long expectedLength = count * index.Dimensions;
        if (originalVectors.Length < expectedLength)
        {
            throw new ArgumentException(
                $"originalVectors too small. Expected {expectedLength}, got {originalVectors.Length}.");
        }

        if (residualVectors.Length < expectedLength)
        {
            throw new ArgumentException(
                $"residualVectors too small. Expected {expectedLength}, got {residualVectors.Length}.");
        }

        unsafe
        {
            fixed (float* pX = originalVectors)
            fixed (float* pRes = residualVectors)
            fixed (long* pKeys = keys)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_compute_residual_n(
                        index.Handle, count, pX, pRes, pKeys)
                );
            }
        }
    }
}