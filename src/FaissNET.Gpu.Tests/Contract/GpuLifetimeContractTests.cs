using System.Runtime.CompilerServices;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class GpuLifetimeContractTests
{
    private const int Stored = 32;

    /// <summary>A device index shares ownership of its provider's resources, so it stays usable after the provider is disposed.</summary>
    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void DeviceIndex_OutlivesDisposedProvider(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: 4, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var expected = SearchAssert.Search(gpu, queries, 3);

        provider.Dispose();

        var actual = SearchAssert.Search(gpu, queries, 3);
        Assert.Equal(expected.Labels, actual.Labels);
        Assert.Equal(expected.Distances, actual.Distances);

        if (c.Has(GpuCapabilities.DeviceAdd))
        {
            ((IIDSequentialFloatIndex)gpu).Add(4, Vectors.Random(seed: 3, count: 4, dimensions: c.Dimensions));
            Assert.Equal(Stored + 4, gpu.TotalCount);
        }

        using var back = c.ToCpu(gpu);
        Assert.Equal(gpu.TotalCount, back.TotalCount);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void DeviceIndex_OutlivesDisposedCpuSource(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: 4, dimensions: c.Dimensions);
        var cpu = c.CreatePopulatedCpuIndex(stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, 0);
        var expected = SearchAssert.Search(gpu, queries, 3);

        cpu.Dispose();

        var actual = SearchAssert.Search(gpu, queries, 3);
        Assert.Equal(expected.Labels, actual.Labels);
        Assert.Equal(Stored, gpu.TotalCount);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void CpuCopy_OutlivesDeviceIndexAndProvider(GpuIndexCase c)
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions);
        var queries = Vectors.Random(seed: 2, count: 4, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(stored);
        var provider = new GpuResourcesProvider();
        var gpu = c.ToGpu(provider, cpu, 0);
        using var back = c.ToCpu(gpu);
        var expected = SearchAssert.Search(back, queries, 3);

        gpu.Dispose();
        provider.Dispose();

        var actual = SearchAssert.Search(back, queries, 3);
        Assert.Equal(expected.Labels, actual.Labels);
        Assert.Equal(expected.Distances, actual.Distances);
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void DisposedDeviceIndex_ThrowsObjectDisposed(GpuIndexCase c)
    {
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        var gpu = c.ToGpu(provider, cpu, 0);

        gpu.Dispose();
        gpu.Dispose();

        Assert.Throws<ObjectDisposedException>(() => gpu.TotalCount);
        Assert.Throws<ObjectDisposedException>(() => SearchAssert.Search(gpu, new float[c.Dimensions], 1));
        Assert.Throws<ObjectDisposedException>(() => c.ToCpu(gpu));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Transfer_DisposedCpuSource_ThrowsObjectDisposed(GpuIndexCase c)
    {
        var cpu = c.CreateCpuIndex();
        cpu.Dispose();
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        Assert.Throws<ObjectDisposedException>(() => c.ToGpu(provider, cpu, 0));
        Assert.Throws<ObjectDisposedException>(() => c.ToGpuWithOptions(provider, cpu, options, 0));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Transfer_DisposedProvider_ThrowsObjectDisposed(GpuIndexCase c)
    {
        using var cpu = c.CreateCpuIndex();
        var provider = new GpuResourcesProvider();
        provider.Dispose();
        using var options = new GpuClonerOptions();

        Assert.Throws<ObjectDisposedException>(() => c.ToGpu(provider, cpu, 0));
        Assert.Throws<ObjectDisposedException>(() => c.ToGpuWithOptions(provider, cpu, options, 0));
    }

    [Theory]
    [MemberData(nameof(GpuIndexCatalog.All), MemberType = typeof(GpuIndexCatalog))]
    public void Transfer_NullArguments_ThrowArgumentNull(GpuIndexCase c)
    {
        using var cpu = c.CreateCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        Assert.Throws<ArgumentNullException>(() => c.ToGpu(null!, cpu, 0));
        Assert.Throws<ArgumentNullException>(() => c.ToGpuWithOptions(null!, cpu, options, 0));
        Assert.Throws<ArgumentNullException>(() => c.ToCpu(null!));
    }

    [Fact]
    public void Transfer_NullCpuIndex_ThrowsArgumentNull()
    {
        using var provider = new GpuResourcesProvider();

        Assert.Throws<ArgumentNullException>(() => GpuIndexProvider.TransferToGpu(provider, (IndexFlatL2)null!));
    }

    [Fact]
    public void DisposedProvider_EveryMemberThrowsObjectDisposed()
    {
        var provider = new GpuResourcesProvider();
        provider.Dispose();
        provider.Dispose();

        Assert.Throws<ObjectDisposedException>(() => provider.SetTempMemory(1024));
        Assert.Throws<ObjectDisposedException>(() => provider.SetPinnedMemory(1024));
        Assert.Throws<ObjectDisposedException>(provider.NoTempMemory);
        Assert.Throws<ObjectDisposedException>(() => provider.SetDefaultStream(0, IntPtr.Zero));
        Assert.Throws<ObjectDisposedException>(provider.SetDefaultNullStreamAllDevices);
        Assert.Throws<ObjectDisposedException>(() => provider.GetResources());
    }

    /// <summary>
    /// <see cref="GpuResources"/> points into the provider's native resources. O
    /// nce the provider is disposed every member must throw instead of dereferencing freed memory.
    /// </summary>
    [Fact]
    public void Resources_AfterProviderDisposed_ThrowObjectDisposed()
    {
        var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();
        resources.InitializeForDevice(0);

        provider.Dispose();

        for (var member = 0; member < ResourcesMemberCount; member++)
        {
            var thrown = false;
            try
            {
                InvokeResourcesMember(resources, member);
            }
            catch (ObjectDisposedException)
            {
                thrown = true;
            }

            Assert.True(thrown, $"GpuResources member #{member} did not throw ObjectDisposedException.");
        }
    }

    /// <summary>Undisposed device indexes and providers are finalized in any order without freeing shared state twice.</summary>
    [Fact]
    public void Finalization_OfUndisposedDeviceIndexesAndProviders_IsSafe()
    {
        var stored = Vectors.Random(seed: 1, count: Stored, dimensions: 8);
        var queries = Vectors.Random(seed: 2, count: 4, dimensions: 8);

        for (var i = 0; i < 8; i++)
        {
            Abandon(stored);
        }

        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();

        using var cpu = new IndexFlatL2(8);
        cpu.Add(Stored, stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu);
        var (distances, labels) = SearchAssert.Search(gpu, queries, 3);
        SearchAssert.MatchesBruteForce(Faiss.Models.MetricType.L2, stored, queries, 3, distances, labels);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Abandon(float[] stored)
    {
        var cpu = new IndexFlatL2(8);
        cpu.Add(Stored, stored);
        var provider = new GpuResourcesProvider();
        var gpu = GpuIndexProvider.TransferToGpu(provider, cpu);
        _ = GpuIndexProvider.TransferToCpu(gpu);
    }

    private const int ResourcesMemberCount = 10;

    private static void InvokeResourcesMember(GpuResources resources, int member)
    {
        switch (member)
        {
            case 0: resources.InitializeForDevice(0); break;
            case 1: resources.GetBlasHandle(0); break;
            case 2: resources.GetBlasHandleCurrentDevice(); break;
            case 3: resources.GetDefaultStream(0); break;
            case 4: resources.GetDefaultStreamCurrentDevice(); break;
            case 5: resources.GetAsyncCopyStream(0); break;
            case 6: resources.GetAsyncCopyStreamCurrentDevice(); break;
            case 7: resources.GetPinnedMemory(out _, out _); break;
            case 8: resources.SyncDefaultStream(0); break;
            case 9: resources.SyncDefaultStreamCurrentDevice(); break;
            default: throw new ArgumentOutOfRangeException(nameof(member));
        }
    }
}
