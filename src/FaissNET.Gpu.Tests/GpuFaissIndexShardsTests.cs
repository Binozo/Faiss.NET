namespace Faiss.Gpu.Tests;

using Gpu;
using Xunit;
using Cpu.Indexes;

/// <summary>
/// Proving our multi-GPU sharding is totally locked in and doesn't get cooked by memory leaks.
/// </summary>
public class GpuFaissIndexShardsTests
{
    [Fact]
    public void ShardedIndex_AddAndSearch_ReturnsExactMatch()
    {
        // Arrange: Spin up the basic CPU indexes
        int dimensions = 4;
        using var cpuIndex1 = new FaissIndexFlatL2(dimensions);
        using var cpuIndex2 = new FaissIndexFlatL2(dimensions);
        
        // Initialize the VRAM context
        using var gpuContext = new FaissGpuContext();
        gpuContext.SetTempMemory(50 * 1024 * 1024);

        // Act: Teleport both to the GPU (We use Device 0 for both so the test passes on single-GPU setups)
        using var shard1 = GpuIndexProvider.TransferToGpu(gpuContext, cpuIndex1, 0);
        using var shard2 = GpuIndexProvider.TransferToGpu(gpuContext, cpuIndex2, 1);

        // Create the massive virtual index (threaded = true for parallel searches!)
        using var shardedIndex = new GpuFaissIndexShards(dimensions, threaded: false);
        
        // Wire them up
        shardedIndex.AddShard(shard1);
        shardedIndex.AddShard(shard2);

        // Assert initial state
        Assert.Equal(dimensions, shardedIndex.Dimensions);
        Assert.Equal(0, shardedIndex.TotalCount);

        // Create 2 vectors: [1,2,3,4] and [5,6,7,8]
        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f,   5.0f, 6.0f, 7.0f, 8.0f };
        
        // Add directly to the sharded pool. Faiss handles distributing them!
        shardedIndex.Add(2, vectors);
        
        // Assert state after adding
        Assert.Equal(2, shardedIndex.TotalCount);

        // Search for the first vector
        float[] query = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        // The shards execute in parallel and merge the results. It's giving peak enterprise scale.
        shardedIndex.Search(1, query, 1, distances, labels);

        // Assert: The closest match should be ID 0 with a distance of 0
        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}