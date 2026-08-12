using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Distances;

public static class Fvec
{
    /// <summary>
    /// Compute one queryVector against vectors (vector => vectors) inner products.
    /// </summary>
    public static void InnerProducts(Span<float> innerProducts, ReadOnlySpan<float> queryVector, int count, Span<float> vectors, int dimension)
    {
        if (Decimal.IsNegative(dimension) || dimension == 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "must be a positive integer.");
        
        if (Decimal.IsNegative(count) || count == 0)
            throw new ArgumentOutOfRangeException(nameof(count), count, "must be a positive integer.");

        if (innerProducts.Length != count)
            throw new ArgumentOutOfRangeException(nameof(innerProducts), innerProducts.Length, "array must be count");

        if (queryVector.Length != dimension)
            throw new ArgumentOutOfRangeException(nameof(queryVector), queryVector.Length, "array length must be dimension");

        if (vectors.Length % dimension != 0)
            throw new ArgumentOutOfRangeException(nameof(vectors), vectors.Length, "array length must be divisible by dimension");
        
        Native.faiss_fvec_inner_products_ny(innerProducts, queryVector, vectors, (nuint)dimension, (nuint)count);
    }
    
    /// <summary>
    /// Compute count square L2 distance between x and a set of contiguous y vectors
    /// </summary>
    public static void L2Sqr(Span<float> squaredDistances, ReadOnlySpan<float> queryVector, int count, Span<float> vectors, int dimension)
    {
        if (Decimal.IsNegative(dimension) || dimension == 0)
            throw new ArgumentOutOfRangeException(nameof(dimension), dimension, "must be a positive integer.");
        
        if (Decimal.IsNegative(count) || count == 0)
            throw new ArgumentOutOfRangeException(nameof(count), count, "must be a positive integer.");

        if (squaredDistances.Length != count)
            throw new ArgumentOutOfRangeException(nameof(squaredDistances), squaredDistances.Length, "array must be count");

        if (queryVector.Length != dimension)
            throw new ArgumentOutOfRangeException(nameof(queryVector), queryVector.Length, "array length must be dimension");

        if (vectors.Length % dimension != 0)
            throw new ArgumentOutOfRangeException(nameof(vectors), vectors.Length, "array length must be divisible by dimension");
        
        Native.faiss_fvec_L2sqr_ny(squaredDistances, queryVector, vectors, (nuint)dimension, (nuint)count);
    }

    /// <summary>
    /// Calculates the squared norm of a vector
    /// </summary>
    public static float NormL2Sqr(ReadOnlySpan<float> vectors, int dimensions)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "must be a positive integer.");

        if (vectors.Length % dimensions != 0)
            throw new ArgumentOutOfRangeException(nameof(vectors), "array must be divisible by dimensions");
        
        return Native.faiss_fvec_norm_L2sqr(vectors, (nuint)dimensions);
    }
    
    /// <summary>
    /// Compute the L2 norms for a set of vectors
    /// </summary>
    /// <param name="norms">The output norm for each vector (norms.Length = nx)</param>
    /// <param name="count">Count of total vectors concatenated in the vectors span</param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static void NormsL2(Span<float> norms, ReadOnlySpan<float> vectors, int dimensions, int count)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "must be a positive integer.");

        if (vectors.Length / dimensions != count)
            throw new ArgumentOutOfRangeException(nameof(vectors), "array must be equal to count * dimensions");
        
        Native.faiss_fvec_norms_L2(norms, vectors, (nuint)dimensions, (nuint)count);
    }
    
    /// <summary>
    /// Compute the squared L2 norms for a set of vectors
    /// </summary>
    /// <param name="norms">The output norm for each vector (norms.Length = nx)</param>
    /// <param name="count">Count of total vectors concatenated in the vectors span</param>
    /// <returns></returns>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static void NormsL2Sqr(Span<float> norms, ReadOnlySpan<float> vectors, int dimensions, int count)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "must be a positive integer.");

        if (vectors.Length / dimensions != count)
            throw new ArgumentOutOfRangeException(nameof(vectors), "array must be equal to count * dimensions");
        
        Native.faiss_fvec_norms_L2sqr(norms, vectors, (nuint)dimensions, (nuint)count);
    }
    
    /// <summary>
    /// L2-renormalize a set of vector. Nothing done if the vector is 0-normed
    /// </summary>
    /// <param name="count">Count of total vectors concatenated in the vectors span</param>
    public static void RenormL2(int dimensions, int count, Span<float> vectors)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), "must be a positive integer.");

        if (Decimal.IsNegative(count) || count == 0)
            throw new ArgumentOutOfRangeException(nameof(count), "must be a positive integer.");

        if (vectors.Length / dimensions != count)
            throw new ArgumentOutOfRangeException(nameof(vectors), "array must be equal to count * dimensions");
        
        Native.faiss_fvec_renorm_L2((nuint)dimensions, (nuint)count, vectors);
    }
}