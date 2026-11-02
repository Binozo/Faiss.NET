using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Resources;
using Faiss.Gpu.Tests.Infrastructure;
using Faiss.Tests.Infrastructure;
using Xunit;

namespace Faiss.Gpu.Tests.Contract;

[Collection(GpuTestCollection.Name)]
public class GpuDeviceContractTests
{
    private const int Stored = 64;
    private const int K = 5;

    public static TheoryData<GpuIndexCase, int> CasesOnEveryDevice() => OnEveryDevice(GpuCapabilities.None);

    public static TheoryData<GpuIndexCase, int> DeviceAddableOnEveryDevice() => OnEveryDevice(GpuCapabilities.DeviceAdd);

    public static TheoryData<int> InvalidDevices { get; } = new(-1, FaissGpu.GetNumGpus(), int.MaxValue, int.MinValue);

    [Theory]
    [MemberData(nameof(CasesOnEveryDevice))]
    public void TransferToDevice_SearchMatchesCpuSource(GpuIndexCase c, int device)
    {
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        using var plain = c.ToGpu(provider, cpu, device);
        using var withOptions = c.ToGpuWithOptions(provider, cpu, options, device);

        SearchAssert.SameResults(plain, withOptions, queries, K, relTol: 0, absTol: 0);
        using var back = c.ToCpu(plain);
        SearchAssert.SameResults(plain, back, queries, K, relTol: c.DistanceTolerance);
    }

    [Theory]
    [MemberData(nameof(DeviceAddableOnEveryDevice))]
    public async Task UntrainedOnDevice_TrainsAddsAndSearches(GpuIndexCase c, int device)
    {
        using var cpu = c.CreateUntrainedCpuIndex();
        using var provider = new GpuResourcesProvider();
        using var gpu = c.ToGpu(provider, cpu, device);

        await ((ITrainableFloatIndex)gpu).TrainAsync(c.TrainCount, c.TrainingVectors());
        ((IIDSequentialFloatIndex)gpu).Add(Stored, Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));

        Assert.Equal(Enumerable.Range(0, Stored).Select(i => (long)i), SearchAssert.StoredLabels(gpu, Stored));
    }

    /// <summary>One provider serves several devices at once, each with its own copy of the data.</summary>
    [Fact]
    public void OneProvider_ServesEveryDeviceSideBySide()
    {
        var c = GpuIndexCatalog.Get("IVFFlatL2");
        var queries = Vectors.Random(seed: 2, count: 8, dimensions: c.Dimensions);
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: Stored, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        var copies = Enumerable.Range(0, FaissGpu.GetNumGpus()).Select(device => c.ToGpu(provider, cpu, device)).ToArray();

        try
        {
            ((IIDSequentialFloatIndex)copies[^1]).Add(1, new float[c.Dimensions]);

            Assert.Equal(Stored + 1, copies[^1].TotalCount);
            Assert.All(copies[..^1], copy => Assert.Equal(Stored, copy.TotalCount));
            Assert.All(copies[..^1], copy => SearchAssert.SameResults(cpu, copy, queries, K));
        }
        finally
        {
            foreach (var copy in copies)
            {
                copy.Dispose();
            }
        }
    }

    [Theory]
    [MemberData(nameof(InvalidDevices))]
    public void Transfer_InvalidDevice_ThrowsArgumentOutOfRange(int device)
    {
        var c = GpuIndexCatalog.Get("FlatL2");
        using var cpu = c.CreatePopulatedCpuIndex(Vectors.Random(seed: 1, count: 4, dimensions: c.Dimensions));
        using var provider = new GpuResourcesProvider();
        using var options = new GpuClonerOptions();

        var plain = Assert.Throws<ArgumentOutOfRangeException>(() => c.ToGpu(provider, cpu, device));
        var withOptions = Assert.Throws<ArgumentOutOfRangeException>(() => c.ToGpuWithOptions(provider, cpu, options, device));

        Assert.Equal("deviceId", plain.ParamName);
        Assert.Equal("deviceId", withOptions.ParamName);
        using var valid = c.ToGpu(provider, cpu, 0);
        Assert.Equal(4, valid.TotalCount);
    }

    private static TheoryData<GpuIndexCase, int> OnEveryDevice(GpuCapabilities caps)
    {
        var data = new TheoryData<GpuIndexCase, int>();
        foreach (var c in GpuIndexCatalog.With(caps))
        {
            for (var device = 0; device < FaissGpu.GetNumGpus(); device++)
            {
                data.Add(c, device);
            }
        }

        return data;
    }
}
