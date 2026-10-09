using Faiss.Cpu.Clustering;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Exceptions;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Tests.Clustering;

public class ClusteringTests
{
    private const int Dimensions = 4;
    private const int K = 8;
    private const int Count = 512;

    [Fact]
    public void Constructor_ExposesShapeAndFaissDefaults()
    {
        using var clustering = new Cpu.Clustering.Clustering(Dimensions, K);

        Assert.Equal(Dimensions, clustering.Dimensions);
        Assert.Equal(K, clustering.K);
        Assert.Equal(25, clustering.Niter);
        Assert.Equal(1, clustering.Nredo);
        Assert.Equal(39, clustering.MinPointsPerCentroid);
        Assert.Equal(256, clustering.MaxPointsPerCentroid);
        Assert.Equal(1234, clustering.Seed);
    }

    [Fact]
    public void DefaultOptions_MatchTheParameterlessConstructor()
    {
        using var plain = new Faiss.Cpu.Clustering.Clustering(Dimensions, K);
        using var configured = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, new ClusteringOptions());

        Assert.Equal(plain.Niter, configured.Niter);
        Assert.Equal(plain.Nredo, configured.Nredo);
        Assert.Equal(plain.MinPointsPerCentroid, configured.MinPointsPerCentroid);
        Assert.Equal(plain.MaxPointsPerCentroid, configured.MaxPointsPerCentroid);
        Assert.Equal(plain.Seed, configured.Seed);
    }

    [Fact]
    public void Train_WithDefaultOptions_ProducesKCentroids()
    {
        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, new ClusteringOptions());
        using var index = new IndexFlatL2(Dimensions);

        var train = Vectors.Random(seed: 150, count: Count, dimensions: Dimensions);
        clustering.Train(Count, train, index);

        var centroids = clustering.GetCentroids();

        Assert.Equal(K * Dimensions, centroids.Length);

        Assert.All(centroids, v => Assert.InRange(v, -1.5f, 1.5f));
        Assert.Contains(centroids, v => v != 0f);

        Assert.Equal(K, index.TotalCount);
    }

    [Fact]
    public void Train_AssignsEveryTrainingVectorToItsNearestCentroid()
    {
        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, new ClusteringOptions { Niter = 10 });
        using var index = new IndexFlatL2(Dimensions);

        var train = Vectors.Random(seed: 151, count: Count, dimensions: Dimensions);
        clustering.Train(Count, train, index);

        var centroids = clustering.GetCentroids();

        for (var i = 0; i < 16; i++)
        {
            var query = train.AsSpan(i * Dimensions, Dimensions);

            var distances = new float[1];
            var labels = new long[1];
            index.Search(1, query, 1, distances, labels);

            var best = 0;
            var bestScore = Oracles.L2Sqr(query, centroids.AsSpan(0, Dimensions));
            for (var c = 1; c < K; c++)
            {
                var score = Oracles.L2Sqr(query, centroids.AsSpan(c * Dimensions, Dimensions));
                if (score < bestScore)
                {
                    best = c;
                    bestScore = score;
                }
            }

            Assert.Equal(best, labels[0]);
            FloatAssert.Equal(bestScore, distances[0], relTol: 1e-4f, absTol: 1e-5f);
        }
    }

    [Fact]
    public void Train_IsDeterministicForAFixedSeed()
    {
        var train = Vectors.Random(seed: 152, count: Count, dimensions: Dimensions);

        var first = Cluster(train, seed: 7);
        var second = Cluster(train, seed: 7);
        var different = Cluster(train, seed: 99);

        FloatAssert.Equal(first, second);
        Assert.NotEqual(first, different);
    }

    [Fact]
    public void Options_AreForwardedToTheNativeClustering()
    {
        var options = new ClusteringOptions
        {
            Niter = 7,
            Nredo = 2,
            Spherical = true,
            Seed = 4242,
            MinPointsPerCentroid = 3,
            MaxPointsPerCentroid = 64,
        };

        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, options);

        Assert.Equal(7, clustering.Niter);
        Assert.Equal(2, clustering.Nredo);
        Assert.True(clustering.Spherical);
        Assert.Equal(4242, clustering.Seed);
        Assert.Equal(3, clustering.MinPointsPerCentroid);
        Assert.Equal(64, clustering.MaxPointsPerCentroid);
    }

    [Fact]
    public void Train_FewerPointsThanCentroids_Throws()
    {
        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, new ClusteringOptions());
        using var index = new IndexFlatL2(Dimensions);

        var train = Vectors.Random(seed: 153, count: K - 1, dimensions: Dimensions);

        Assert.Throws<FaissNativeException>(() => clustering.Train(K - 1, train, index));
    }

    [Fact]
    public void GetCentroids_BeforeTraining_IsEmpty()
    {
        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K);

        Assert.Empty(clustering.GetCentroids());
    }

    [Fact]
    public void UseAfterDispose_Throws()
    {
        var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K);
        clustering.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = clustering.K);
    }

    private static float[] Cluster(float[] train, int seed)
    {
        using var clustering = new Faiss.Cpu.Clustering.Clustering(Dimensions, K, new ClusteringOptions { Seed = seed, Niter = 5 });
        using var index = new IndexFlatL2(Dimensions);

        clustering.Train(Count, train, index);

        return clustering.GetCentroids();
    }
}
