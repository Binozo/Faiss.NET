using Faiss.Cpu.Indexes;
using Faiss.Exceptions;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Interfaces;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class ReconstructContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_ReturnsAddedVectors(IndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);
        Sequential(index).Add(2, vectors);
        c.EnableReconstruct(index);

        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        FloatAssert.Equal(vectors.AsSpan(0, c.Dimensions), reconstruct.Reconstruct(0));
        FloatAssert.Equal(vectors.AsSpan(c.Dimensions, c.Dimensions), reconstruct.Reconstruct(1));
        FloatAssert.Equal(vectors, reconstruct.Reconstruct(0, 2));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableBinary), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_ReturnsAddedVectors_Binary(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        var vectors = Vectors.RandomBinary(seed: 1, count: 2, bitDimensions: c.Dimensions);
        Sequential(index).Add(2, vectors);

        var reconstruct = Assert.IsAssignableFrom<IReconstructBinaryIndex>(index);

        Assert.Equal(vectors.AsSpan(0, c.CodeSize).ToArray(), reconstruct.Reconstruct(0));
        Assert.Equal(vectors.AsSpan(c.CodeSize, c.CodeSize).ToArray(), reconstruct.Reconstruct(1));
        Assert.Equal(vectors, reconstruct.Reconstruct(0, 2));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_NegativeKey_Throws(IndexCase c)
    {
        using var index = c.CreateIndex();
        Sequential(index).Add(2, Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions));

        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        Assert.Throws<ArgumentOutOfRangeException>(() => reconstruct.Reconstruct(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => reconstruct.Reconstruct(0, -2));
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_BatchPastTheEnd_ThrowsInsteadOfAborting(IndexCase c)
    {
        using var index = c.CreateIndex();
        Sequential(index).Add(2, Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions));

        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        AssertRejected(Record.Exception(() => reconstruct.Reconstruct(1, 2)));
        AssertRejected(Record.Exception(() => reconstruct.Reconstruct(2, 1)));
        Assert.Equal(c.Dimensions, reconstruct.Reconstruct(1, 1).Length);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableBinary), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_PastTheEnd_Throws_Binary(BinaryIndexCase c)
    {
        using var index = c.CreateIndex();
        Sequential(index).Add(2, Vectors.RandomBinary(seed: 1, count: 2, bitDimensions: c.Dimensions));

        var reconstruct = Assert.IsAssignableFrom<IReconstructBinaryIndex>(index);

        AssertRejected(Record.Exception(() => reconstruct.Reconstruct(2)));
        AssertRejected(Record.Exception(() => reconstruct.Reconstruct(1_000_000)));
        AssertRejected(Record.Exception(() => reconstruct.Reconstruct(1, 2)));
        Assert.Equal(c.CodeSize, reconstruct.Reconstruct(1, 1).Length);
    }

    private static void AssertRejected(Exception? exception) => Assert.True(exception is ArgumentOutOfRangeException or FaissNativeException, $"Expected a rejection, got {exception?.GetType().Name ?? "no exception"}.");

    [Theory]
    [MemberData(nameof(IndexCatalog.NearestLossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_Lossy_IsNearestToItsOwnSourceVector(IndexCase c)
    {
        const int count = 16;

        using var index = c.CreateIndex();
        var vectors = Vectors.Random(seed: 240, count: count, dimensions: c.Dimensions);
        Sequential(index).Add(count, vectors);
        c.EnableReconstruct(index);

        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        for (var i = 0; i < count; i++)
        {
            var restored = reconstruct.Reconstruct(i);

            Assert.Equal(c.Dimensions, restored.Length);

            // The reconstruction must be closer to the vector it came from than to any other vector
            // in the set as equality cant be checked by the loss of precision through quantization.
            var own = Oracles.L2Sqr(restored, vectors.AsSpan(i * c.Dimensions, c.Dimensions));
            for (var j = 0; j < count; j++)
            {
                if (j == i)
                    continue;

                Assert.True(
                    own <= Oracles.L2Sqr(restored, vectors.AsSpan(j * c.Dimensions, c.Dimensions)),
                    $"{c.Name}: reconstruction of vector {i} is closer to vector {j}."
                );
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.LossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Reconstruct_Lossy_BatchMatchesSingleKeys(IndexCase c)
    {
        const int count = 8;

        using var index = c.CreateIndex();
        Sequential(index).Add(count, Vectors.Random(seed: 241, count: count, dimensions: c.Dimensions));
        c.EnableReconstruct(index);

        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);
        var batch = reconstruct.Reconstruct(0, count);

        Assert.Equal(count * c.Dimensions, batch.Length);

        for (var i = 0; i < count; i++)
        {
            FloatAssert.Equal(reconstruct.Reconstruct(i), batch.AsSpan(i * c.Dimensions, c.Dimensions));
        }
    }

    private static IIDSequentialFloatIndex Sequential(FloatIndex index) => Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index);

    private static IIDSequentialBinaryIndex Sequential(BinaryIndex index) => Assert.IsAssignableFrom<IIDSequentialBinaryIndex>(index);
}
