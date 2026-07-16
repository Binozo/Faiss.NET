using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Distances;

public static class Fvec
{
    /// <summary>
    /// Calculates the squared norm of a vector
    /// </summary>
    public static float NormL2Sqr(ReadOnlySpan<float> vectors, int dimensions)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "dimensions must be a positive integer.");

        if (vectors.Length % dimensions != 0)
            throw new ArgumentOutOfRangeException(nameof(vectors), "vector array must be divisible by dimensions");
        
        return Native.faiss_fvec_norm_L2sqr(vectors, (nuint)dimensions);
    }
    
    /// <summary>
    /// L2-renormalize a set of vector. Nothing done if the vector is 0-normed
    /// </summary>
    /// <param name="count">Count of total vectors concatenated in the vectors span</param>
    public static void RenormL2(int dimensions, int count, Span<float> vectors)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "dimensions must be a positive integer.");

        if (Decimal.IsNegative(count) || count == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "count must be a positive integer.");

        if (vectors.Length / dimensions != count)
            throw new ArgumentOutOfRangeException(nameof(count), "vector array must be equal to count * dimensions");
        
        Native.faiss_fvec_renorm_L2((nuint)dimensions, (nuint)count, vectors);
    }
}