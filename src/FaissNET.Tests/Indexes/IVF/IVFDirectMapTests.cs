using Faiss.Models;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Indexes.IVF;

public class IVFDirectMapTests
{
    private const int Dimensions = 8;
    private const int BinaryDimensions = 64;

    private static readonly long[] GappedIds = [0, 1, 2, 5];
    private static readonly float[] Source = Vectors.Random(seed: 904, count: GappedIds.Length, dimensions: Dimensions);

    public sealed record IvfCase(string Name, Func<Task<FloatIndex>> Create, Action<FloatIndex, bool> SetDirectMap)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<IvfCase> Cases { get; } =
    [
        new IvfCase("IVFFlat", async () =>
            {
                var index = new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(Dimensions), Dimensions, nlist: 2, ownQuantizer: true);
                await index.TrainAsync(64, Vectors.Random(seed: 901, count: 64, dimensions: Dimensions));
                return index;
            },
            (i, on) => ((IndexIVFFlat<IndexFlatL2>)i).DirectMap = on),
        new IvfCase("IVFPQ", async () =>
            {
                var index = new IndexIVFPQ(Dimensions, n: 2, m: 4);
                await index.TrainAsync(512, Vectors.Random(seed: 902, count: 512, dimensions: Dimensions));
                return index;
            },
            (i, on) => ((IndexIVFPQ)i).DirectMap = on),
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task SingleKeyReconstruct_NeedsAnExplicitDirectMap(IvfCase c)
    {
        using var index = await Populated(c, ids: null);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(0));

        c.SetDirectMap(index, true);
        Assert.Equal(Dimensions, reconstruct.Reconstruct(0).Length);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Reconstruct_DoesNotEnableTheDirectMap_SoRemoveIdsKeepsWorking(IvfCase c)
    {
        using var index = await Populated(c, ids: null);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        reconstruct.Reconstruct(0, 2);
        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(0));

        Assert.Equal(1, Assert.IsAssignableFrom<IIDRemovableFloatIndex>(index).RemoveIds(IDSelector.Range(0, 1)));
        Assert.Equal(GappedIds.Length - 1, index.TotalCount);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task BatchReconstruct_WorksWithNonSequentialIds(IvfCase c)
    {
        using var index = await Populated(c, GappedIds);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        var batch = reconstruct.Reconstruct(0, 3);

        for (var i = 0; i < 3; i++)
        {
            var row = batch.AsSpan(i * Dimensions, Dimensions);
            var own = Oracles.L2Sqr(row, Source.AsSpan(i * Dimensions, Dimensions));
            for (var j = 0; j < GappedIds.Length; j++)
            {
                Assert.True(j == i || own < Oracles.L2Sqr(row, Source.AsSpan(j * Dimensions, Dimensions)), $"{c.Name}: row {i} is nearer to vector {j}.");
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task FailedDirectMap_LeavesTheIndexAcceptingIds(IvfCase c)
    {
        using var index = await Populated(c, GappedIds);

        Assert.Throws<FaissNativeException>(() => c.SetDirectMap(index, true));

        Assert.IsAssignableFrom<IIDMappedFloatIndex>(index).Add(1, Vectors.Random(seed: 903, count: 1, dimensions: Dimensions), [6]);
        Assert.Equal(GappedIds.Length + 1, index.TotalCount);
    }

    [Fact]
    public async Task Binary_SingleKeyReconstruct_NeedsAnExplicitDirectMap()
    {
        using var index = await PopulatedBinary(ids: null);

        Assert.Throws<FaissNativeException>(() => index.Reconstruct(0));

        index.MakeDirectMap(true);
        Assert.Equal(index.CodeSize, index.Reconstruct(0).Length);
    }

    [Fact]
    public async Task Binary_Reconstruct_DoesNotEnableTheDirectMap_SoRemoveIdsKeepsWorking()
    {
        using var index = await PopulatedBinary(ids: null);

        index.Reconstruct(0, 2);

        Assert.Equal(1, index.RemoveIds(IDSelector.Range(0, 1)));
    }

    [Fact]
    public async Task Binary_BatchReconstruct_WorksWithNonSequentialIds()
    {
        var vectors = Vectors.RandomBinary(seed: 905, count: GappedIds.Length, bitDimensions: BinaryDimensions);
        using var index = await PopulatedBinary(GappedIds, vectors);

        Assert.Equal(vectors.AsSpan(0, 3 * index.CodeSize).ToArray(), index.Reconstruct(0, 3));
    }

    [Fact]
    public async Task Binary_FailedDirectMap_LeavesTheIndexAcceptingIds()
    {
        using var index = await PopulatedBinary(GappedIds);

        Assert.Throws<FaissNativeException>(() => index.MakeDirectMap(true));

        index.Add(1, Vectors.RandomBinary(seed: 906, count: 1, bitDimensions: BinaryDimensions), [6]);
        Assert.Equal(GappedIds.Length + 1, index.TotalCount);
    }

    [Fact]
    public async Task Binary_FailedDirectMapType_LeavesTheIndexAcceptingIds()
    {
        using var index = await PopulatedBinary(GappedIds);

        Assert.Throws<FaissNativeException>(() => index.DirectMapType = DirectMapType.Array);

        index.Add(1, Vectors.RandomBinary(seed: 909, count: 1, bitDimensions: BinaryDimensions), [6]);
        Assert.Equal(GappedIds.Length + 1, index.TotalCount);
    }

    [Fact]
    public async Task Binary_HashtableDirectMap_ReconstructsNonSequentialIds()
    {
        var vectors = Vectors.RandomBinary(seed: 910, count: GappedIds.Length, bitDimensions: BinaryDimensions);
        using var index = await PopulatedBinary(GappedIds, vectors);

        index.DirectMapType = DirectMapType.Hashtable;

        Assert.Equal(vectors.AsSpan(3 * index.CodeSize, index.CodeSize).ToArray(), index.Reconstruct(5));
    }

    private static async Task<FloatIndex> Populated(IvfCase c, long[]? ids)
    {
        var index = await c.Create();
        var vectors = Source;
        if (ids is null)
            Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(GappedIds.Length, vectors);
        else
            Assert.IsAssignableFrom<IIDMappedFloatIndex>(index).Add(ids.Length, vectors, ids);

        return index;
    }

    private static async Task<IndexBinaryIVF> PopulatedBinary(long[]? ids, byte[]? vectors = null)
    {
        var index = new IndexBinaryIVF(BinaryDimensions, nlist: 2, hnswM: null);
        await index.TrainAsync(64, Vectors.RandomBinary(seed: 907, count: 64, bitDimensions: BinaryDimensions));
        vectors ??= Vectors.RandomBinary(seed: 908, count: GappedIds.Length, bitDimensions: BinaryDimensions);
        if (ids is null)
            index.Add(GappedIds.Length, vectors);
        else
            index.Add(ids.Length, vectors, ids);

        return index;
    }
}
