using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Gpu.Cloning;

namespace Faiss.Gpu.Tests;

using Xunit;

using Cpu.Indexes;
using Gpu.Resources;

/// <summary>
/// Tests multi-GPU transfer scenarios with <see cref="GpuMultipleClonerOptions"/>
/// and <see cref="GpuIndexProvider.TransferToGpuMultiple{T}"/>.
/// </summary>
[Collection("GpuSequential")]
public class GpuMultipleGpuTests
{
    [Fact]
    public void GpuMultipleClonerOptions_Shard_DefaultIsFalse()
    {
        using var options = new GpuMultipleClonerOptions();
        Assert.False(options.Shard);
    }

    [Fact]
    public void GpuMultipleClonerOptions_Shard_CanBeEnabled()
    {
        using var options = new GpuMultipleClonerOptions();
        options.Shard = true;
        Assert.True(options.Shard);
    }

    [Fact]
    public void GpuMultipleClonerOptions_ShardType_DefaultIsIdModulo()
    {
        using var options = new GpuMultipleClonerOptions();
        Assert.Equal(GpuShardType.IdModulo, options.ShardType);
    }

    [Fact]
    public void GpuMultipleClonerOptions_ShardType_CanBeChanged()
    {
        using var options = new GpuMultipleClonerOptions();
        options.ShardType = GpuShardType.IdRange;
        Assert.Equal(GpuShardType.IdRange, options.ShardType);
    }

    [Fact]
    public void GpuMultipleClonerOptions_InheritsGpuClonerOptionsProperties()
    {
        using var options = new GpuMultipleClonerOptions
        {
            UseFloat16 = true,
            UsePrecomputed = true,
            StorageMode = IndicesOptions.Bit64,
            Shard = true,
            ShardType = GpuShardType.InvertedList
        };

        Assert.True(options.UseFloat16);
        Assert.True(options.UsePrecomputed);
        Assert.Equal(IndicesOptions.Bit64, options.StorageMode);
        Assert.True(options.Shard);
        Assert.Equal(GpuShardType.InvertedList, options.ShardType);
    }

    [Fact]
    public void TransferToGpuMultiple_SingleDevice_Succeeds()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);
        cpuIndex.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        using var resources = new GpuResourcesProvider();
        var contexts = new[] { resources };
        var devices = new[] { 0 };

        using var sharded = GpuIndexProvider.TransferToGpuMultiple(contexts, devices, cpuIndex);

        Assert.Single(sharded.Devices);
        Assert.Equal(0, sharded.Devices[0]);
        Assert.Equal(dimensions, sharded.Dimensions);
    }

    [Fact]
    public void TransferToGpuMultiple_WithOptions_SingleDevice_Succeeds()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var cpuIndex = new IndexFlatL2(dimensions);
        cpuIndex.Add([1.0f, 2.0f, 3.0f, 4.0f]);

        using var resources = new GpuResourcesProvider();
        using var options = new GpuMultipleClonerOptions { Shard = false };
        var contexts = new[] { resources };
        var devices = new[] { 0 };

        using var sharded = GpuIndexProvider.TransferToGpuMultiple(contexts, devices, cpuIndex, options);

        Assert.Single(sharded.Devices);
    }

    [Fact]
    public void TransferToGpuMultiple_NullArguments_ThrowsArgumentNullException()
    {
        using var index = new IndexFlatL2(4);
        using var resources = new GpuResourcesProvider();
        using var options = new GpuMultipleClonerOptions();
        var contexts = new[] { resources };
        var devices = new[] { 0 };

        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple<IndexFlatL2>(null!, devices, index, options));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple(contexts, null!, index, options));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple<IndexFlatL2>(contexts, devices, null!, options));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple<IndexFlatL2>(contexts, devices, index, null!));
    }

    [Fact]
    public void TransferToGpuMultiple_EmptyContexts_ThrowsArgumentException()
    {
        using var index = new IndexFlatL2(4);
        using var options = new GpuMultipleClonerOptions();
        var emptyContexts = Array.Empty<GpuResourcesProvider>();
        var devices = new[] { 0 };

        Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuMultiple(emptyContexts, devices, index, options));
    }

    [Fact]
    public void TransferToGpuMultiple_MismatchedLengths_ThrowsArgumentException()
    {
        using var index = new IndexFlatL2(4);
        using var resources1 = new GpuResourcesProvider();
        using var options = new Cloning.GpuMultipleClonerOptions();
        var contexts = new[] { resources1 };
        var devices = new[] { 0, 1 };

        Assert.Throws<ArgumentException>(() => GpuIndexProvider.TransferToGpuMultiple(contexts, devices, index, options));
    }

    [Fact]
    public void TransferToGpuMultiple_DefaultOptions_NullArguments_ThrowsArgumentNullException()
    {
        using var index = new IndexFlatL2(4);
        using var resources = new GpuResourcesProvider();
        var contexts = new[] { resources };
        var devices = new[] { 0 };

        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple<IndexFlatL2>(null!, devices, index));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple(contexts, null!, index));
        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpuMultiple<IndexFlatL2>(contexts, devices, null!));
    }
}
