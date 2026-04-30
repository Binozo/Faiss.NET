namespace Faiss.Gpu.Tests;

using Xunit;

using Gpu.Resources;

/// <summary>
/// Tests the global GPU utilities exposed by <see cref="FaissGpu"/>
/// and the <see cref="GpuResourcesProvider"/> lifecycle.
/// </summary>
[Collection("GpuSequential")]
public class GpuResourcesTests
{
    [Fact]
    public void GetNumGpus_ReturnsNonNegative()
    {
        GpuFact.SkipIfNoGpu();

        int count = FaissGpu.GetNumGpus();
        Assert.True(count >= 1);
    }

    [Fact]
    public void GpuResourcesProvider_CanBeConstructedAndDisposed()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        Assert.NotNull(resources);
    }

    [Fact]
    public void GpuResourcesProvider_SetTempMemory_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        resources.SetTempMemory(16 * 1024 * 1024);
    }

    [Fact]
    public void GpuResourcesProvider_SetTempMemory_NegativeValue_ThrowsArgumentOutOfRange()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        Assert.Throws<ArgumentOutOfRangeException>(() => resources.SetTempMemory(-1));
    }

    [Fact]
    public void GpuResourcesProvider_SetPinnedMemory_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        resources.SetPinnedMemory(8 * 1024 * 1024);
    }

    [Fact]
    public void GpuResourcesProvider_NoTempMemory_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        resources.NoTempMemory();
    }

    [Fact]
    public void GpuResourcesProvider_SetDefaultStream_WithZero_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        resources.SetDefaultStream(0, System.IntPtr.Zero);
    }

    [Fact]
    public void GpuResourcesProvider_SetDefaultNullStreamAllDevices_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        resources.SetDefaultNullStreamAllDevices();
    }

    [Fact]
    public void GpuResourcesProvider_GetResources_ReturnsValidHandle()
    {
        GpuFact.SkipIfNoGpu();

        using var resources = new GpuResourcesProvider();
        var gpuRes = resources.GetResources();

        Assert.True(gpuRes.GetDefaultStream(0) != System.IntPtr.Zero);
    }

    [Fact]
    public void SyncAllDevices_DoesNotThrow()
    {
        GpuFact.SkipIfNoGpu();

        FaissGpu.SyncAllDevices();
    }
}
