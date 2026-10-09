using Faiss.Cpu.Distances;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Distances;

public class FvecTests
{
    private const int Dimensions = 4;
    private const int Count = 6;

    [Fact]
    public void InnerProducts_MatchesOraclePerVector()
    {
        var query = Vectors.Random(seed: 60, count: 1, dimensions: Dimensions);
        var vectors = Vectors.Random(seed: 61, count: Count, dimensions: Dimensions);
        var results = new float[Count];

        Fvec.InnerProducts(results, query, Count, vectors, Dimensions);

        for (var i = 0; i < Count; i++)
        {
            FloatAssert.Equal(Oracles.InnerProduct(query, vectors.AsSpan(i * Dimensions, Dimensions)), results[i]);
        }
    }

    [Fact]
    public void L2Sqr_MatchesSquaredOraclePerVector()
    {
        var query = Vectors.Random(seed: 60, count: 1, dimensions: Dimensions);
        var vectors = Vectors.Random(seed: 61, count: Count, dimensions: Dimensions);
        var results = new float[Count];

        Fvec.L2Sqr(results, query, Count, vectors, Dimensions);

        for (var i = 0; i < Count; i++)
        {
            // Faiss returns the squared distance, never the root.
            FloatAssert.Equal(Oracles.L2Sqr(query, vectors.AsSpan(i * Dimensions, Dimensions)), results[i]);
        }
    }

    [Fact]
    public void L2Sqr_SelfDistanceIsZero()
    {
        var vectors = Vectors.Random(seed: 62, count: 1, dimensions: Dimensions);
        var results = new float[1];

        Fvec.L2Sqr(results, vectors, 1, vectors, Dimensions);

        FloatAssert.Equal(0f, results[0]);
    }

    [Fact]
    public void NormL2Sqr_EqualsSelfInnerProduct()
    {
        var vector = Vectors.Random(seed: 63, count: 1, dimensions: Dimensions);

        FloatAssert.Equal(Oracles.InnerProduct(vector, vector), Fvec.NormL2Sqr(vector, Dimensions));
    }

    [Fact]
    public void NormL2Sqr_RejectsAnythingOtherThanOneVector()
    {
        var vectors = Vectors.Random(seed: 63, count: 3, dimensions: Dimensions);

        // The native call norms only the first vector, so accepting a batch here would silently
        // discard the rest; callers must slice or use NormsL2Sqr.
        Assert.Throws<ArgumentException>(() => Fvec.NormL2Sqr(vectors, Dimensions));
        Assert.Throws<ArgumentException>(() => Fvec.NormL2Sqr(vectors.AsSpan(0, Dimensions - 1), Dimensions));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.NormL2Sqr(vectors.AsSpan(0, Dimensions), 0));

        // Slicing a single vector out of the batch is the supported way to norm one row.
        var second = vectors.AsSpan(Dimensions, Dimensions);
        FloatAssert.Equal(Oracles.InnerProduct(second, second), Fvec.NormL2Sqr(second, Dimensions));
    }

    [Fact]
    public void NormsL2_AreTheRootsOfTheSquaredNorms()
    {
        var vectors = Vectors.Random(seed: 64, count: Count, dimensions: Dimensions);
        var norms = new float[Count];
        var squared = new float[Count];

        Fvec.NormsL2(norms, vectors, Dimensions, Count);
        Fvec.NormsL2Sqr(squared, vectors, Dimensions, Count);

        for (var i = 0; i < Count; i++)
        {
            var expected = Oracles.InnerProduct(vectors.AsSpan(i * Dimensions, Dimensions), vectors.AsSpan(i * Dimensions, Dimensions));

            FloatAssert.Equal(expected, squared[i]);
            FloatAssert.Equal(MathF.Sqrt(expected), norms[i]);
        }
    }

    [Fact]
    public void RenormL2_ScalesEveryVectorToUnitLength()
    {
        var vectors = Vectors.Random(seed: 65, count: Count, dimensions: Dimensions);
        var original = vectors.ToArray();

        Fvec.RenormL2(Dimensions, Count, vectors);

        var norms = new float[Count];
        Fvec.NormsL2(norms, vectors, Dimensions, Count);

        for (var i = 0; i < Count; i++)
        {
            FloatAssert.Equal(1f, norms[i], relTol: 1e-4f, absTol: 1e-5f);

            // Direction is preserved: the renormalized vector is the original divided by its norm.
            var scale = MathF.Sqrt(Oracles.InnerProduct(original.AsSpan(i * Dimensions, Dimensions), original.AsSpan(i * Dimensions, Dimensions)));
            for (var j = 0; j < Dimensions; j++)
            {
                FloatAssert.Equal(original[(i * Dimensions) + j] / scale, vectors[(i * Dimensions) + j], relTol: 1e-4f, absTol: 1e-5f);
            }
        }
    }

    [Fact]
    public void RenormL2_LeavesZeroVectorUntouched()
    {
        var vectors = new float[Dimensions * 2];
        vectors[Dimensions] = 3f;

        Fvec.RenormL2(Dimensions, 2, vectors);

        // Documented behaviour: a zero-normed vector is left alone rather than producing NaN.
        Assert.Equal(new float[Dimensions], vectors[..Dimensions]);
        FloatAssert.Equal(1f, vectors[Dimensions]);
    }

    [Fact]
    public void InnerProducts_InvalidArguments_Throw()
    {
        var query = new float[Dimensions];
        var vectors = new float[Count * Dimensions];

        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.InnerProducts(new float[Count], query, Count, vectors, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.InnerProducts(new float[Count], query, 0, vectors, Dimensions));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.InnerProducts(new float[Count - 1], query, Count, vectors, Dimensions));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.InnerProducts(new float[Count], new float[Dimensions - 1], Count, vectors, Dimensions));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.InnerProducts(new float[Count], query, Count, new float[(Count * Dimensions) - 1], Dimensions));
    }

    [Fact]
    public void L2Sqr_InvalidArguments_Throw()
    {
        var query = new float[Dimensions];
        var vectors = new float[Count * Dimensions];

        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.L2Sqr(new float[Count], query, Count, vectors, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.L2Sqr(new float[Count - 1], query, Count, vectors, Dimensions));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.L2Sqr(new float[Count], new float[Dimensions + 1], Count, vectors, Dimensions));
    }

    [Fact]
    public void Norms_UndersizedOutput_Throws()
    {
        var vectors = new float[Count * Dimensions];

        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.NormsL2(new float[Count - 1], vectors, Dimensions, Count));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.NormsL2Sqr(new float[Count - 1], vectors, Dimensions, Count));
    }

    [Fact]
    public void RenormL2_InvalidArguments_Throw()
    {
        var vectors = new float[Count * Dimensions];

        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.RenormL2(0, Count, vectors));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.RenormL2(Dimensions, 0, vectors));
        Assert.Throws<ArgumentOutOfRangeException>(() => Fvec.RenormL2(Dimensions, Count + 1, vectors));
    }
}
