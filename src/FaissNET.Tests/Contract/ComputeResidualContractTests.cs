using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Contract;

public class ComputeResidualContractTests
{
    private const int Dimensions = 4;
    private const int Count = 6;

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Single_ResidualIsInputMinusReconstruction(IndexCase c)
    {
        using var index = Populated(c);
        var residual = Residual(index);
        var reconstruct = Assert.IsAssignableFrom<IReconstructFloatIndex>(index);
        var query = Vectors.Random(seed: 171, count: 1, dimensions: c.Dimensions);

        for (long key = 0; key < Count; key++)
        {
            var actual = new float[c.Dimensions];
            residual.ComputeResidual(query, actual, key);

            var stored = reconstruct.Reconstruct(key);
            for (var j = 0; j < c.Dimensions; j++)
            {
                FloatAssert.Equal(query[j] - stored[j], actual[j]);
            }
        }
    }

    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Batch_MatchesRepeatedSingleCalls(IndexCase c)
    {
        using var index = Populated(c);
        var residual = Residual(index);

        var queries = Vectors.Random(seed: 172, count: 3, dimensions: c.Dimensions);
        long[] keys = [4, 0, 2];

        var batch = new float[3 * c.Dimensions];
        residual.ComputeResidual(queries, batch, keys);

        for (var i = 0; i < keys.Length; i++)
        {
            var single = new float[c.Dimensions];
            residual.ComputeResidual(queries.AsSpan(i * c.Dimensions, c.Dimensions), single, keys[i]);

            FloatAssert.Equal(single, batch.AsSpan(i * c.Dimensions, c.Dimensions));
        }
    }

    /// <summary>
    /// faiss's compute_residual_n rejects a bad key inside an OpenMP region, which aborts the process; every index must
    /// surface it as an exception. Two keys, because a single-key batch was observed not to abort even unguarded.
    /// </summary>
    [Theory]
    [MemberData(nameof(IndexCatalog.ReconstructableFloat), MemberType = typeof(IndexCatalog))]
    [MemberData(nameof(IndexCatalog.LossyReconstructableFloat), MemberType = typeof(IndexCatalog))]
    public void Batch_OutOfRangeKey_ThrowsInsteadOfAbortingInOpenMp(IndexCase c)
    {
        using var index = Populated(c);
        var residual = Residual(index);

        var queries = Vectors.Random(seed: 173, count: 2, dimensions: c.Dimensions);
        var residuals = new float[2 * c.Dimensions];

        var exception = Record.Exception(() => residual.ComputeResidual(queries, residuals, [0L, (long)Count]));
        Assert.True(exception is ArgumentOutOfRangeException or FaissNativeException, $"{c.Name}: got {exception?.GetType().Name ?? "no exception"}.");
    }

    private static FloatIndex Populated(IndexCase c)
    {
        var index = c.CreateIndex();
        Assert.IsAssignableFrom<IIDSequentialFloatIndex>(index).Add(Count, Vectors.Random(seed: 170, count: Count, dimensions: c.Dimensions));
        c.EnableReconstruct(index);
        return index;
    }

    private static IComputeResidualFloatIndex Residual(FloatIndex index) => Assert.IsAssignableFrom<IComputeResidualFloatIndex>(index);

    [Fact]
    public void Single_OutOfRangeKey_Throws()
    {
        using var index = Create(out _);

        var query = new float[Dimensions];
        var residual = new float[Dimensions];

        Assert.Throws<ArgumentOutOfRangeException>(() => index.ComputeResidual(query, residual, -1));
        Assert.Throws<FaissNativeException>(() => index.ComputeResidual(query, residual, Count));
    }

    [Fact]
    public void InvalidSpanLengths_Throw()
    {
        using var index = Create(out _);

        Assert.Throws<ArgumentException>(() => index.ComputeResidual(new float[Dimensions - 1], new float[Dimensions], 0));
        Assert.Throws<ArgumentException>(() => index.ComputeResidual(new float[Dimensions], new float[Dimensions + 1], 0));
        Assert.Throws<ArgumentException>(() => index.ComputeResidual(new float[Dimensions], new float[2 * Dimensions], [0L, 1L]));
        Assert.Throws<ArgumentException>(() => index.ComputeResidual(new float[2 * Dimensions], new float[Dimensions], [0L, 1L]));
    }

    private static IndexFlatL2 Create(out float[] vectors)
    {
        var index = new IndexFlatL2(Dimensions);
        vectors = Vectors.Random(seed: 170, count: Count, dimensions: Dimensions);
        index.Add(Count, vectors);

        return index;
    }
}
