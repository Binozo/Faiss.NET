using Faiss.Cpu.Transforms;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Transforms;

public class VectorTransformTests
{
    private const int Dimensions = 8;
    private const int Count = 64;

    [Fact]
    public void Normalization_ScalesEveryVectorToUnitLength()
    {
        using var transform = new NormalizationTransform(Dimensions);

        Assert.True(transform.IsTrained);
        Assert.Equal(Dimensions, transform.DIn);
        Assert.Equal(Dimensions, transform.DOut);
        Assert.Equal(2.0f, transform.Norm);

        var vectors = Vectors.Random(seed: 140, count: 4, dimensions: Dimensions);
        var applied = transform.Apply(4, vectors);

        Assert.Equal(4 * Dimensions, applied.Length);

        for (var i = 0; i < 4; i++)
        {
            var row = applied.AsSpan(i * Dimensions, Dimensions);
            FloatAssert.Equal(1f, MathF.Sqrt(Oracles.InnerProduct(row, row)), relTol: 1e-4f, absTol: 1e-5f);
        }
    }

    [Fact]
    public void Normalization_UnsupportedNorm_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NormalizationTransform(Dimensions, norm: 1.0f));
    }

    [Fact]
    public void Centering_SubtractsTheTrainedMeanAndReversesExactly()
    {
        using var transform = new CenteringTransform(Dimensions);
        Assert.False(transform.IsTrained);

        var train = Vectors.Random(seed: 141, count: Count, dimensions: Dimensions);
        transform.Train(Count, train);
        Assert.True(transform.IsTrained);

        var mean = new float[Dimensions];
        for (var i = 0; i < Count; i++)
        {
            for (var j = 0; j < Dimensions; j++)
            {
                mean[j] += train[(i * Dimensions) + j] / Count;
            }
        }

        var centeredMean = transform.Apply(1, mean);
        for (var j = 0; j < Dimensions; j++)
        {
            FloatAssert.Equal(0f, centeredMean[j], relTol: 1e-4f, absTol: 1e-5f);
        }

        var sample = train.AsSpan(0, Dimensions).ToArray();
        var applied = transform.Apply(1, sample);
        var round = new float[Dimensions];
        transform.ReverseTransform(applied, round);

        FloatAssert.Equal(sample, round, relTol: 1e-4f, absTol: 1e-5f);
    }

    [Fact]
    public void Centering_ApplyBeforeTraining_Throws()
    {
        using var transform = new CenteringTransform(Dimensions);

        Assert.Throws<FaissUntrainedException>(() => transform.Apply(1, new float[Dimensions]));
    }

    [Fact]
    public void RandomRotation_PreservesPairwiseL2Distances()
    {
        using var transform = new RandomRotationMatrix(Dimensions, Dimensions);

        var vectors = Vectors.Random(seed: 142, count: 8, dimensions: Dimensions);
        transform.Train(8, vectors);
        Assert.True(transform.IsTrained);

        var rotated = transform.Apply(8, vectors);

        for (var i = 0; i < 8; i++)
        {
            for (var j = i + 1; j < 8; j++)
            {
                var before = Oracles.L2Sqr(vectors.AsSpan(i * Dimensions, Dimensions), vectors.AsSpan(j * Dimensions, Dimensions));
                var after = Oracles.L2Sqr(rotated.AsSpan(i * Dimensions, Dimensions), rotated.AsSpan(j * Dimensions, Dimensions));

                FloatAssert.Equal(before, after, relTol: 1e-4f, absTol: 1e-4f);
            }
        }
    }

    [Fact]
    public void RandomRotation_ReverseRecoversTheInput()
    {
        using var transform = new RandomRotationMatrix(Dimensions, Dimensions);
        var vectors = Vectors.Random(seed: 143, count: 2, dimensions: Dimensions);
        transform.Train(2, vectors);

        var rotated = transform.Apply(2, vectors);
        var round = new float[2 * Dimensions];
        transform.ReverseTransform(rotated, round);

        FloatAssert.Equal(vectors, round, relTol: 1e-3f, absTol: 1e-4f);
    }

    [Fact]
    public void RemapDimensions_NonUniform_CopiesLeadingDimensionsAndZeroFills()
    {
        const int dOut = Dimensions + 3;

        using var transform = new RemapDimensionsTransform(Dimensions, dOut);

        Assert.Equal(Dimensions, transform.DIn);
        Assert.Equal(dOut, transform.DOut);

        var vectors = Vectors.Random(seed: 144, count: 1, dimensions: Dimensions);
        var padded = transform.Apply(1, vectors);

        Assert.Equal(dOut, padded.Length);
        FloatAssert.Equal(vectors, padded.AsSpan(0, Dimensions));

        for (var j = Dimensions; j < dOut; j++)
        {
            FloatAssert.Equal(0f, padded[j]);
        }
    }

    [Fact]
    public void Pca_ReducesDimensionalityAndPreservesNeighbourOrdering()
    {
        const int dOut = 4;

        using var transform = new PCAMatrix(Dimensions, dOut);
        Assert.False(transform.IsTrained);
        Assert.Equal(0f, transform.EigenPower);
        Assert.True(transform.IsReversible);

        var train = Vectors.Random(seed: 145, count: 256, dimensions: Dimensions);
        transform.Train(256, train);

        Assert.True(transform.IsTrained);
        Assert.Equal(dOut, transform.DOut);

        var projected = transform.Apply(4, train);
        Assert.Equal(4 * dOut, projected.Length);
    }

    [Fact]
    public void Pca_Whitening_IsNotReversible()
    {
        using var transform = new PCAMatrix(Dimensions, 4, eigenPower: -0.5f);

        Assert.False(transform.IsReversible);
        Assert.Throws<NotSupportedException>(() => transform.ReverseTransform(new float[4], new float[Dimensions]));
    }

    [Fact]
    public void Itq_IsNotReversible()
    {
        using var matrix = new ITQMatrix(Dimensions);
        using var pipeline = new ITQTransform(Dimensions, Dimensions, doPca: false);

        Assert.False(matrix.IsReversible);
        Assert.False(pipeline.IsReversible);

        Assert.Throws<NotSupportedException>(() => matrix.ReverseTransform(new float[Dimensions], new float[Dimensions]));
        Assert.Throws<NotSupportedException>(() => pipeline.ReverseTransform(new float[Dimensions], new float[Dimensions]));
    }

    [Fact]
    public void ItqTransform_TrainsAndAppliesToTheRequestedWidth()
    {
        using var transform = new ITQTransform(Dimensions, Dimensions, doPca: false);

        var train = Vectors.Random(seed: 146, count: 256, dimensions: Dimensions);
        transform.Train(256, train);

        Assert.True(transform.IsTrained);

        var applied = transform.Apply(2, train);
        Assert.Equal(2 * Dimensions, applied.Length);
    }

    [Fact]
    public void Opq_TrainsAndPreservesWidth()
    {
        using var transform = new OPQMatrix(Dimensions, m: 4, d2: -1);

        Assert.Equal(50, transform.Niter);
        transform.Niter = 5;
        Assert.Equal(5, transform.Niter);

        var train = Vectors.Random(seed: 147, count: 512, dimensions: Dimensions);
        transform.Train(512, train);

        Assert.True(transform.IsTrained);
        Assert.Equal(Dimensions, transform.DOut);

        var applied = transform.Apply(2, train);
        Assert.Equal(2 * Dimensions, applied.Length);
    }

    [Fact]
    public void Apply_InvalidArguments_Throw()
    {
        using var transform = new NormalizationTransform(Dimensions);

        Assert.Throws<ArgumentOutOfRangeException>(() => transform.Apply(-1, new float[Dimensions]));
        Assert.Throws<ArgumentException>(() => transform.Apply(2, new float[Dimensions]));
    }

    [Fact]
    public void Train_InvalidArguments_Throw()
    {
        using var transform = new CenteringTransform(Dimensions);

        Assert.Throws<ArgumentOutOfRangeException>(() => transform.Train(-1, new float[Dimensions]));
        Assert.Throws<ArgumentException>(() => transform.Train(4, new float[Dimensions]));
    }

    [Fact]
    public void ReverseTransform_InvalidBuffers_Throw()
    {
        using var transform = new CenteringTransform(Dimensions);
        transform.Train(Count, Vectors.Random(seed: 148, count: Count, dimensions: Dimensions));

        Assert.Throws<ArgumentException>(() => transform.ReverseTransform(new float[Dimensions - 1], new float[Dimensions]));
        Assert.Throws<ArgumentException>(() => transform.ReverseTransform(Array.Empty<float>(), new float[Dimensions]));
        Assert.Throws<ArgumentException>(() => transform.ReverseTransform(new float[Dimensions], new float[Dimensions - 1]));
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var transform = new NormalizationTransform(Dimensions);
        transform.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = transform.DIn);
    }
}
