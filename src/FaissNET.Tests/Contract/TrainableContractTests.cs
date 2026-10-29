using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class TrainableContractTests
{
    [Theory]
    [MemberData(nameof(IndexCatalog.TrainableFloat), MemberType = typeof(IndexCatalog))]
    public void Add_BeforeTrain_Throws(IndexCase c)
    {
        using var index = c.CreateUntrainedIndex();
        var trainable = Assert.IsAssignableFrom<ITrainableFloatIndex>(index);
        Assert.False(trainable.IsTrained);

        var vectors = Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions);
        Assert.Throws<FaissUntrainedException>(() => Sequential(index).Add(4, vectors));
        Assert.Equal(0, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.TrainableFloat), MemberType = typeof(IndexCatalog))]
    public async Task Train_ThenAdd_SelfQueryRank1(IndexCase c)
    {
        Assert.True(c.TrainCount > 0);

        using var index = c.CreateUntrainedIndex();
        var trainable = Assert.IsAssignableFrom<ITrainableFloatIndex>(index);

        var train = Vectors.Random(seed: 1, count: c.TrainCount, dimensions: c.Dimensions);
        await trainable.TrainAsync(c.TrainCount, train);
        Assert.True(trainable.IsTrained);
        Assert.Equal(0, index.TotalCount);

        if (index is IIVFIndex ivf)
            ivf.Nprobe = ivf.Nlist;

        var added = Vectors.Random(seed: 2, count: 4, dimensions: c.Dimensions);
        Sequential(index).Add(4, added);
        Assert.Equal(4, index.TotalCount);

        var query = added.AsSpan(0, c.Dimensions);
        Span<float> distances = stackalloc float[1];
        Span<long> labels = stackalloc long[1];
        index.Search(1, query, k: 1, distances, labels);
        Assert.Equal(0, labels[0]);
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.TrainableFloat), MemberType = typeof(IndexCatalog))]
    public void TrainAsync_InvalidArguments_ThrowSynchronously(IndexCase c)
    {
        using var index = c.CreateUntrainedIndex();
        var trainable = Assert.IsAssignableFrom<ITrainableFloatIndex>(index);
        var vectors = Vectors.Random(seed: 1, count: 2, dimensions: c.Dimensions);

        // Throw is sync as the guards are checking before a Task is returned thus Ignore() is never reached
        Assert.Throws<ArgumentOutOfRangeException>(() => Ignore(trainable.TrainAsync(-1, vectors)));
        Assert.Throws<ArgumentException>(() => Ignore(trainable.TrainAsync(4, vectors)));
        Assert.False(trainable.IsTrained);
    }

    private static void Ignore(Task task)
    {
    }

    private static IIDSequentialFloatIndex Sequential(FloatIndex index) => Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index);
}
