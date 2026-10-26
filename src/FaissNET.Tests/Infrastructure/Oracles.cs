using System.Numerics;

namespace Faiss.Tests.Infrastructure;

public static class Oracles
{
    /// <summary>
    /// Squared Euclidean distance which Faiss L2 indexes return for instance.
    /// </summary>
    public static float L2Sqr(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Length mismatch: {a.Length} vs {b.Length}.");
        }

        double sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            sum += d * d;
        }

        return (float)sum;
    }

    /// <summary>
    /// Inner product which Faiss IP indexes return for instance.
    /// </summary>
    public static float InnerProduct(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Length mismatch: {a.Length} vs {b.Length}.");
        }

        double sum = 0;
        for (var i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }

        return (float)sum;
    }

    /// <summary>
    /// Hamming distance on packed bits.
    /// </summary>
    public static int Hamming(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"Length mismatch: {a.Length} vs {b.Length}.");
        }

        var distance = 0;
        for (var i = 0; i < a.Length; i++)
        {
            distance += BitOperations.PopCount((uint)(a[i] ^ b[i]));
        }

        return distance;
    }
}
