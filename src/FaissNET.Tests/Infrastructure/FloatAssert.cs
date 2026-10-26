using Xunit;

namespace Faiss.Tests.Infrastructure;

public static class FloatAssert
{
    public static void Equal(float expected, float actual, float relTol = 1e-5f, float absTol = 1e-6f)
        => EqualCore(expected, actual, relTol, absTol, index: null);

    public static void Equal(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float relTol = 1e-5f, float absTol = 1e-6f)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            EqualCore(expected[i], actual[i], relTol, absTol, i);
        }
    }

    private static void EqualCore(float expected, float actual, float relTol, float absTol, int? index)
    {
        var diff = Math.Abs(expected - actual);
        var tol = Math.Max(absTol, relTol * Math.Max(Math.Abs(expected), Math.Abs(actual)));

        if (diff <= tol)
            return;

        var prefix = index is int i ? $"Index {i}: " : string.Empty;
        Assert.Fail($"{prefix}Expected {expected:G9}, actual {actual:G9}, |diff|={diff:G9} (tol={tol:G9})");
    }
}
