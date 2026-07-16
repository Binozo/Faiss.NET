using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Distances;

public static class Pairwise
{
    /// <summary>
    /// Compute pairwise distances between sets of vectors
    /// </summary>
    /// <param name="ldq">Leading dimension of queryVectors. Defaults to -1 => dimensions</param>
    /// <param name="ldb">Leading dimension of vectors. Defaults to -1 => dimensions</param>
    /// <param name="ldd">Leading dimension of distanceMatrix. Defaults to -1 => count</param>
    /// <exception cref="ArgumentOutOfRangeException"></exception>
    public static void L2Sqr(long dimensions, long queryVectorsCount, ReadOnlySpan<float> queryVectors, long count, ReadOnlySpan<float> vectors, Span<float> distanceMatrix, long ldq = -1, long ldb = -1, long ldd = -1)
    {
        if (Decimal.IsNegative(dimensions) || dimensions == 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "must be a positive integer.");

        if (queryVectors.Length / dimensions != queryVectorsCount)
            throw new ArgumentOutOfRangeException(nameof(queryVectors), queryVectors.Length / dimensions, "array must be of length queryVectorsCount * dimensions");

        if (vectors.Length / dimensions != count)
            throw new ArgumentOutOfRangeException(nameof(vectors), vectors.Length / dimensions, "array must be of length count * dimensions");

        if (distanceMatrix.Length != queryVectorsCount * count)
            throw new ArgumentOutOfRangeException(nameof(distanceMatrix), distanceMatrix.Length, "array must be of length queryVectorsCount * count");

        Native.faiss_pairwise_L2sqr(dimensions, queryVectorsCount, queryVectors, count, vectors, distanceMatrix, ldq, ldb, ldd);
    }
}