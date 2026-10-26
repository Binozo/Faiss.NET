namespace Faiss.Tests.Infrastructure;

public static class Vectors
{
    public static float[] Random(int seed, int count, int dimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);

        var rng = new Random(seed);
        var values = new float[count * dimensions];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        return values;
    }

    public static float[] Filled(int count, int dimensions, float value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimensions);

        var values = new float[count * dimensions];
        Array.Fill(values, value);
        return values;
    }

    public static byte[] RandomBinary(int seed, int count, int bitDimensions)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        if (bitDimensions <= 0 || bitDimensions % 8 != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(bitDimensions), bitDimensions, "Must be a positive multiple of 8.");
        }

        var values = new byte[count * (bitDimensions / 8)];
        new Random(seed).NextBytes(values);
        return values;
    }
}
