using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Flat;
using Xunit;

namespace Faiss.Tests.Indexes;

/// <summary>
/// Tests for index cloning.
/// </summary>
public class CloneIndexTests
{
    [Fact]
    public void Clone_FloatIndex_CreatesIndependentDeepCopy()
    {
        using var original = new IndexFlatL2(4);
        float[] vector1 = { 1.0f, 1.0f, 1.0f, 1.0f };
        original.Add(1, vector1);

        using var clone = original.Clone();
        Assert.Equal(1, clone.TotalCount);

        float[] vector2 = { 2.0f, 2.0f, 2.0f, 2.0f };
        clone.Add(1, vector2);

        Assert.Equal(1, original.TotalCount);
        Assert.Equal(2, clone.TotalCount);
    }

    [Fact]
    public void Clone_BinaryIndex_CreatesIndependentDeepCopy()
    {
        using var original = new IndexBinaryFlat(16);
        byte[] vector1 = { 255, 0 };
        original.Add(1, vector1);

        using var clone = original.Clone();
        Assert.Equal(1, clone.TotalCount);

        byte[] vector2 = { 0, 255 };
        clone.Add(1, vector2);

        Assert.Equal(1, original.TotalCount);
        Assert.Equal(2, clone.TotalCount);
    }
}
