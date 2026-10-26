using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.Factory;

public class BinaryIndexFactoryTests
{
    private const int Bits = 32;
    private const int CodeSize = Bits / 8;

    [Theory]
    [InlineData("BFlat")]
    [InlineData("BIVF4")]
    [InlineData("BHNSW16")]
    public void Create_BuildsIndexWithRequestedShape(string description)
    {
        using var index = BinaryIndexFactory.Create<GenericBinaryIndex>(description, Bits);

        Assert.Equal(Bits, index.Dimensions);
        Assert.Equal(CodeSize, index.CodeSize);
        Assert.Equal(0, index.TotalCount);
    }

    [Fact]
    public void Create_BFlat_AddAndSelfQuery()
    {
        using var index = BinaryIndexFactory.Create<GenericBinaryIndex>("BFlat", Bits);

        var vectors = Vectors.RandomBinary(seed: 210, count: 8, bitDimensions: Bits);
        index.Add(8, vectors);

        Assert.Equal(8, index.TotalCount);

        var distances = new int[1];
        var labels = new long[1];
        index.Search(1, vectors.AsSpan(0, CodeSize), 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0, distances[0]);
    }

    [Fact]
    public async Task Create_BIVF_RequiresTrainingBeforeAdd()
    {
        using var index = BinaryIndexFactory.Create<GenericBinaryIndex>("BIVF4", Bits);

        Assert.False(index.IsTrained);

        var train = Vectors.RandomBinary(seed: 211, count: 128, bitDimensions: Bits);
        await index.TrainAsync(128, train);

        Assert.True(index.IsTrained);

        var vectors = Vectors.RandomBinary(seed: 212, count: 16, bitDimensions: Bits);
        index.Add(16, vectors);

        Assert.Equal(16, index.TotalCount);
    }

    [Fact]
    public void Create_ConcreteType_IsUsableThroughItsOwnApi()
    {
        using var index = BinaryIndexFactory.Create<IndexBinaryFlat>("BFlat", Bits);

        var vectors = Vectors.RandomBinary(seed: 213, count: 4, bitDimensions: Bits);
        index.Add(4, vectors);

        Assert.Equal(vectors.AsSpan(0, CodeSize).ToArray(), index.Reconstruct(0));
    }

    [Fact]
    public void Create_UnknownDescription_Throws()
    {
        Assert.Throws<FaissNativeException>(() => BinaryIndexFactory.Create<GenericBinaryIndex>("NotABinaryIndex", Bits));
    }

    [Fact]
    public void Create_DimensionsNotMultipleOf8_Throws()
    {
        Assert.ThrowsAny<FaissException>(() => BinaryIndexFactory.Create<GenericBinaryIndex>("BFlat", 12));
    }
}
