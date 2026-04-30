using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Extensions;

/// <summary>
/// Provides standalone encoding and decoding extensions for native Faiss indexes.
/// </summary>
/// <remarks>
/// These methods allow direct compression of vectors to raw bytes and decompression
/// without performing a database search. Not all index types support standalone codecs.
/// </remarks>
public static class IndexCodecExtensions
{
    /// <summary>
    /// Gets the standalone code size in bytes for the specified index.
    /// </summary>
    /// <param name="index">The index to query.</param>
    /// <returns>The number of bytes used to encode a single vector.</returns>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="index"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="FaissException">Thrown when the native call returns an error.</exception>
    public static long GetStandaloneCodeSize(this INativeIndex index)
    {
        ArgumentNullException.ThrowIfNull(index);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_Index_sa_code_size(index.Handle, out nuint size)
        );

        return (long)size;
    }

    /// <summary>
    /// Encodes vectors into raw bytes using the index's standalone codec.
    /// </summary>
    /// <param name="index">The index providing the codec.</param>
    /// <param name="count">The number of vectors to encode.</param>
    /// <param name="vectors">
    /// The input vectors. Length must be at least <paramref name="count"/> * <see cref="IIndex.Dimensions"/>.
    /// </param>
    /// <param name="outputBytes">
    /// The output buffer for encoded bytes. Length must be at least <paramref name="count"/> * <see cref="GetStandaloneCodeSize"/>.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="index"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="count"/> is negative.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="vectors"/> or <paramref name="outputBytes"/> are too small.
    /// </exception>
    /// <exception cref="FaissException">Thrown when the native encoding operation fails.</exception>
    public static void Encode(this INativeIndex index, long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes)
    {
        ArgumentNullException.ThrowIfNull(index);

        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");
        }

        long codeSize = index.GetStandaloneCodeSize();
        long expectedByteLength = count * codeSize;
        long expectedVectorLength = count * index.Dimensions;

        if (vectors.Length < expectedVectorLength)
        {
            throw new ArgumentException(
                $"Input vector span too small. Expected {expectedVectorLength}, got {vectors.Length}.",
                nameof(vectors));
        }

        if (outputBytes.Length < expectedByteLength)
        {
            throw new ArgumentException(
                $"Output byte span too small. Expected {expectedByteLength}, got {outputBytes.Length}.",
                nameof(outputBytes));
        }

        unsafe
        {
            fixed (float* pVectors = vectors)
            fixed (byte* pBytes = outputBytes)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_sa_encode(index.Handle, count, pVectors, pBytes)
                );
            }
        }
    }

    /// <summary>
    /// Decodes raw bytes back into vectors using the index's standalone codec.
    /// </summary>
    /// <param name="index">The index providing the codec.</param>
    /// <param name="count">The number of vectors to decode.</param>
    /// <param name="inputBytes">
    /// The encoded bytes. Length must be at least <paramref name="count"/> * <see cref="GetStandaloneCodeSize"/>.
    /// </param>
    /// <param name="outputVectors">
    /// The output buffer for decoded vectors. Length must be at least <paramref name="count"/> * <see cref="IIndex.Dimensions"/>.
    /// </param>
    /// <exception cref="ArgumentNullException">
    /// Thrown when <paramref name="index"/> is <see langword="null"/>.
    /// </exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="count"/> is negative.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="inputBytes"/> or <paramref name="outputVectors"/> are too small.
    /// </exception>
    /// <exception cref="FaissException">Thrown when the native decoding operation fails.</exception>
    public static void Decode(this INativeIndex index, long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors)
    {
        ArgumentNullException.ThrowIfNull(index);

        if (count < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(count), "Count cannot be negative.");
        }

        long codeSize = index.GetStandaloneCodeSize();
        long expectedByteLength = count * codeSize;
        long expectedVectorLength = count * index.Dimensions;

        if (inputBytes.Length < expectedByteLength)
        {
            throw new ArgumentException(
                $"Input byte span too small. Expected {expectedByteLength}, got {inputBytes.Length}.",
                nameof(inputBytes));
        }

        if (outputVectors.Length < expectedVectorLength)
        {
            throw new ArgumentException(
                $"Output vector span too small. Expected {expectedVectorLength}, got {outputVectors.Length}.",
                nameof(outputVectors));
        }

        unsafe
        {
            fixed (byte* pBytes = inputBytes)
            fixed (float* pVectors = outputVectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_Index_sa_decode(index.Handle, count, pBytes, pVectors)
                );
            }
        }
    }
}
