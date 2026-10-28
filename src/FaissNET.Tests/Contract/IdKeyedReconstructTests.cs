using Faiss.Cpu.Indexes;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Transform;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Serializer;
using Faiss.Cpu.Transforms;
using Faiss.Exceptions;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class IdKeyedReconstructTests
{
    private const int Dimensions = 4;

    private static readonly long[] Ids = [100, 101, 200, 300];
    private static readonly float[] Data = Vectors.Random(seed: 811, count: Ids.Length, dimensions: Dimensions);

    public sealed record IdCase(string Name, Func<FloatIndex> Build)
    {
        public override string ToString() => Name;
    }

    public static TheoryData<IdCase> Cases { get; } =
    [
        new IdCase("IDMap2", () => new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(Dimensions))),
        new IdCase("PreTransform<IDMap2>", () => new IndexPreTransform<IndexIDMap2<IndexFlatL2>>(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(Dimensions)), new NormalizationTransform(Dimensions), ownSubIndex: true)),
        new IdCase("Factory IDMap2,Flat", () => IndexFactory.Create<GenericFloatIndex>("IDMap2,Flat", Dimensions, MetricType.L2)),
        new IdCase("Deserialized PreTransform<IDMap2>", DeserializedPreTransform),
    ];

    [Theory]
    [MemberData(nameof(Cases))]
    public void ReconstructBatch_OverConsecutiveIdsBeyondTheVectorCount_ReturnsTheirVectors(IdCase c)
    {
        using var index = Populated(c);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        var batch = reconstruct.Reconstruct(100, 2);

        FloatAssert.Equal(reconstruct.Reconstruct(100), batch.AsSpan(0, Dimensions));
        FloatAssert.Equal(reconstruct.Reconstruct(101), batch.AsSpan(Dimensions, Dimensions));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ReconstructBatch_OverAMissingId_ThrowsInsteadOfAborting(IdCase c)
    {
        using var index = Populated(c);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);

        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(101, 2)); // 102 does not exist
        Assert.Throws<FaissNativeException>(() => reconstruct.Reconstruct(0, 1));   // 0 does not exist
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ComputeResidualBatch_AcceptsIdsBeyondTheVectorCount(IdCase c)
    {
        using var index = Populated(c);
        var residual = Assert.IsAssignableFrom<IComputeResidualFloatIndex>(index);
        var queries = Vectors.Random(seed: 812, count: 2, dimensions: Dimensions);
        long[] keys = [300, 100];

        var batch = new float[2 * Dimensions];
        residual.ComputeResidual(queries, batch, keys);

        var single = new float[Dimensions];
        for (var i = 0; i < keys.Length; i++)
        {
            residual.ComputeResidual(queries.AsSpan(i * Dimensions, Dimensions), single, keys[i]);
            FloatAssert.Equal(single, batch.AsSpan(i * Dimensions, Dimensions));
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ComputeResidualBatch_MissingIdInsideThePositionRange_ThrowsInsteadOfAborting(IdCase c)
    {
        using var index = Populated(c);
        var residual = Assert.IsAssignableFrom<IComputeResidualFloatIndex>(index);
        var queries = Vectors.Random(seed: 813, count: 2, dimensions: Dimensions);

        // Key 2 passes but no vector has id 2
        Assert.Throws<FaissNativeException>(() => residual.ComputeResidual(queries, new float[2 * Dimensions], [100, 2]));
    }

    [Fact]
    public async Task ComputeResidualBatch_IvfWithoutDirectMap_ThrowsInsteadOfAborting()
    {
        using var quantizer = new IndexFlatL2(Dimensions);
        using var index = new IndexIVFFlat<IndexFlatL2>(quantizer, Dimensions, nlist: 2);
        await index.TrainAsync(64, Vectors.Random(seed: 814, count: 64, dimensions: Dimensions));
        index.Add(Ids.Length, Data);

        Assert.Throws<FaissNativeException>(() => index.ComputeResidual(Data.AsSpan(0, 2 * Dimensions), new float[2 * Dimensions], [0, 1]));
    }

    private static FloatIndex Populated(IdCase c)
    {
        var index = c.Build();
        Assert.IsAssignableFrom<IIDMappedFloatIndex>(index).Add(Ids.Length, Data, Ids);
        return index;
    }

    private static FloatIndex DeserializedPreTransform()
    {
        using var original = new IndexPreTransform<IndexIDMap2<IndexFlatL2>>(new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(Dimensions)), new NormalizationTransform(Dimensions), ownSubIndex: true);
        using var stream = new MemoryStream();
        IndexSerializer.Write(original, stream);
        stream.Position = 0;
        return IndexDeserializer.Read<GenericFloatIndex>(stream);
    }
}
