using Faiss.Cpu.Indexes.Flat;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Models;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Resources;

[Collection(GpuTestCollection.Name)]
public class GpuResourcesTests
{
    public static TheoryData<int> Devices { get; } = new(Enumerable.Range(0, FaissGpu.GetNumGpus()));

    public static TheoryData<int> InvalidDevices { get; } = new(-1, FaissGpu.GetNumGpus(), int.MaxValue);

    [Theory]
    [MemberData(nameof(Devices))]
    public void InitializedDevice_HasItsOwnBlasHandleAndStreams(int device)
    {
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();

        resources.InitializeForDevice(device);

        var blas = resources.GetBlasHandle(device);
        var stream = resources.GetDefaultStream(device);
        var copyStream = resources.GetAsyncCopyStream(device);
        Assert.NotEqual(IntPtr.Zero, blas);
        Assert.NotEqual(IntPtr.Zero, stream);
        Assert.NotEqual(IntPtr.Zero, copyStream);
        Assert.NotEqual(stream, copyStream);

        // Repeated lookups still return the same objects
        Assert.Equal(blas, resources.GetBlasHandle(device));
        Assert.Equal(stream, resources.GetDefaultStream(device));
    }

    [Fact]
    public void Devices_DoNotShareHandlesOrStreams()
    {
        Assert.SkipWhen(FaissGpu.GetNumGpus() < 2, "Needs two GPUs.");
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();

        Assert.NotEqual(resources.GetBlasHandle(0), resources.GetBlasHandle(1));
        Assert.NotEqual(resources.GetDefaultStream(0), resources.GetDefaultStream(1));
        Assert.NotEqual(resources.GetAsyncCopyStream(0), resources.GetAsyncCopyStream(1));
    }

    [Fact]
    public void Providers_DoNotShareStreams()
    {
        using var first = new GpuResourcesProvider();
        using var second = new GpuResourcesProvider();

        Assert.NotEqual(first.GetResources().GetDefaultStream(0), second.GetResources().GetDefaultStream(0));
    }

    /// <summary>The current device is 0 unless something switched it, so the current-device members resolve to device 0.</summary>
    [Fact]
    public void CurrentDeviceMembers_ResolveToDeviceZero()
    {
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();

        Assert.Equal(resources.GetBlasHandle(0), resources.GetBlasHandleCurrentDevice());
        Assert.Equal(resources.GetDefaultStream(0), resources.GetDefaultStreamCurrentDevice());
        Assert.Equal(resources.GetAsyncCopyStream(0), resources.GetAsyncCopyStreamCurrentDevice());
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void SetDefaultStream_Null_AffectsOnlyThatDevice(int device)
    {
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();
        var others = Enumerable.Range(0, FaissGpu.GetNumGpus()).Where(d => d != device).ToDictionary(d => d, d => GetDefaultStream(provider, d));
        resources.InitializeForDevice(device);

        provider.SetDefaultStream(device, IntPtr.Zero);

        Assert.Equal(IntPtr.Zero, resources.GetDefaultStream(device));
        Assert.All(others, other => Assert.Equal(other.Value, GetDefaultStream(provider, other.Key)));
    }

    /// <summary>
    /// Faiss switches the stream of every device from the current one, which aborted the process for any initialized
    /// device other than the current one.
    /// </summary>
    [Fact]
    public void SetDefaultNullStreamAllDevices_AfterInitializingEveryDevice_ResetsAll()
    {
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();
        var count = FaissGpu.GetNumGpus();
        for (var device = 0; device < count; device++)
        {
            resources.InitializeForDevice(device);
        }

        provider.SetDefaultNullStreamAllDevices();

        for (var device = 0; device < count; device++)
        {
            Assert.Equal(IntPtr.Zero, resources.GetDefaultStream(device));
        }

        Assert.Equal(resources.GetBlasHandle(0), resources.GetBlasHandleCurrentDevice());
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void NullDefaultStream_SearchStillCorrect(int device)
    {
        var stored = Vectors.Random(seed: 1, count: 64, dimensions: 8);
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: 8);
        using var cpu = new IndexFlatL2(8);
        cpu.Add(64, stored);
        using var provider = new GpuResourcesProvider();
        provider.GetResources().InitializeForDevice(device);
        provider.SetDefaultNullStreamAllDevices();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, device);

        var (distances, labels) = SearchAssert.Search(gpu, queries, 4);

        SearchAssert.MatchesBruteForce(MetricType.L2, stored, queries, 4, distances, labels);
    }

    [Fact]
    public void PinnedMemory_IsAllocatedAtInitializationWithTheConfiguredSize()
    {
        const int size = 1 << 20;
        using var provider = new GpuResourcesProvider();
        provider.SetPinnedMemory(size);
        var resources = provider.GetResources();

        resources.GetPinnedMemory(out var before, out var beforeSize);
        resources.InitializeForDevice(0);
        resources.GetPinnedMemory(out var after, out var afterSize);

        Assert.Equal(IntPtr.Zero, before);
        Assert.Equal(0u, beforeSize);
        Assert.NotEqual(IntPtr.Zero, after);
        Assert.Equal((nuint)size, afterSize);
    }

    [Fact]
    public void PinnedMemory_Zero_AllocatesNothing()
    {
        using var provider = new GpuResourcesProvider();
        provider.SetPinnedMemory(0);
        var resources = provider.GetResources();

        resources.InitializeForDevice(0);
        resources.GetPinnedMemory(out var ptr, out var size);

        Assert.Equal(IntPtr.Zero, ptr);
        Assert.Equal(0u, size);
    }

    /// <summary>Temporary memory only changes where scratch space comes from.
    /// A pool far too small falls back to device allocation.
    /// </summary>
    [Theory]
    [InlineData(-1L)]
    [InlineData(0L)]
    [InlineData(64L * 1024)]
    [InlineData(64L * 1024 * 1024)]
    public void TempMemory_DoesNotChangeResults(long bytes)
    {
        var stored = Vectors.Random(seed: 1, count: 2000, dimensions: 16);
        var queries = Vectors.Random(seed: 2, count: 300, dimensions: 16);
        using var cpu = new IndexFlatL2(16);
        cpu.Add(2000, stored);
        using var provider = new GpuResourcesProvider();
        if (bytes < 0)
        {
            provider.NoTempMemory();
        }
        else
        {
            provider.SetTempMemory(bytes);
        }

        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu);

        var (distances, labels) = SearchAssert.Search(gpu, queries, 10);
        SearchAssert.MatchesBruteForce(MetricType.L2, stored, queries, 10, distances, labels);
    }

    [Fact]
    public void MemorySizes_Negative_Throw()
    {
        using var provider = new GpuResourcesProvider();

        Assert.Throws<ArgumentOutOfRangeException>(() => provider.SetTempMemory(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => provider.SetPinnedMemory(-1));
    }

    [Theory]
    [MemberData(nameof(InvalidDevices))]
    public void DeviceMembers_InvalidDevice_ThrowArgumentOutOfRange(int device)
    {
        using var provider = new GpuResourcesProvider();
        var resources = provider.GetResources();

        for (var member = 0; member < DeviceMemberCount; member++)
        {
            var thrown = false;
            try
            {
                InvokeDeviceMember(provider, resources, member, device);
            }
            catch (ArgumentOutOfRangeException e)
            {
                Assert.Equal("device", e.ParamName);
                thrown = true;
            }

            Assert.True(thrown, $"Device member #{member} accepted device {device}.");
        }

        Assert.NotEqual(IntPtr.Zero, resources.GetBlasHandle(0));
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void SyncDefaultStream_AfterSearch_LeavesResultsReadable(int device)
    {
        var stored = Vectors.Random(seed: 1, count: 64, dimensions: 8);
        using var cpu = new IndexFlatL2(8);
        cpu.Add(64, stored);
        using var provider = new GpuResourcesProvider();
        using var gpu = GpuIndexProvider.TransferToGpu(provider, cpu, device);
        var (distances, labels) = SearchAssert.Search(gpu, stored, 1);

        provider.GetResources().SyncDefaultStream(device);
        FaissGpu.SyncAllDevices();

        Assert.Equal(Enumerable.Range(0, 64).Select(i => (long)i), labels);
        Assert.All(distances, d => Assert.Equal(0f, d, 1e-5f));
    }

    private const int DeviceMemberCount = 6;

    private static void InvokeDeviceMember(GpuResourcesProvider provider, GpuResources resources, int member, int device)
    {
        switch (member)
        {
            case 0: resources.InitializeForDevice(device); break;
            case 1: resources.GetBlasHandle(device); break;
            case 2: resources.GetDefaultStream(device); break;
            case 3: resources.GetAsyncCopyStream(device); break;
            case 4: resources.SyncDefaultStream(device); break;
            case 5: provider.SetDefaultStream(device, IntPtr.Zero); break;
            default: throw new ArgumentOutOfRangeException(nameof(member));
        }
    }

    private static IntPtr GetDefaultStream(GpuResourcesProvider provider, int device) => provider.GetResources().GetDefaultStream(device);
}
