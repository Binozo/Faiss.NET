using Xunit;

namespace Faiss.Tests;

public interface IFloatAccuracyTest
{
    public void AssertFloatsEqual(float expected, float actual, float relTol = 1e-5f, float absTol = 1e-6f)
    {
        float diff = Math.Abs(expected - actual);
        float tol = Math.Max(absTol, relTol * Math.Max(Math.Abs(expected), Math.Abs(actual)));

        Assert.True(diff <= tol, $"Expected {expected:G9}, actual {actual:G9}, |diff|={diff:G9} (tol={tol:G9})");
    }

    public void AssertFloatsEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float relTol = 1e-5f, float absTol = 1e-6f)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = Math.Abs(e - a);
            float tol = Math.Max(absTol, relTol * Math.Max(Math.Abs(e), Math.Abs(a)));

            Assert.True(diff <= tol, $"Index {i}: expected {e:G9}, actual {a:G9}, |diff|={diff:G9} (tol={tol:G9})");
        }
    }
}