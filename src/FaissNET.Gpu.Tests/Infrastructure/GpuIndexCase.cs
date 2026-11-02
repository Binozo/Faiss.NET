using Faiss.Cpu.Indexes;
using Faiss.Cpu.Interfaces;
using Faiss.Gpu.Cloning;
using Faiss.Gpu.Resources;
using Faiss.Models;
using Faiss.Tests.Infrastructure;

namespace Faiss.Gpu.Tests.Infrastructure;

[Flags]
public enum GpuCapabilities
{
    None = 0,

    /// <summary>The CPU source must be trained before vectors can be added.</summary>
    Train = 1 << 0,

    /// <summary>The device index returns the true nearest neighbours with float32 distances, so it can be checked against a brute-force oracle.</summary>
    Exact = 1 << 1,

    /// <summary>The device wrapper implements <see cref="IIDSequentialFloatIndex"/> and <see cref="IIDMappedFloatIndex"/>.</summary>
    DeviceAdd = 1 << 2,

    /// <summary>The device wrapper implements <see cref="IReconstructFloatIndex"/> and returns the stored vectors bit-for-bit.</summary>
    DeviceReconstruct = 1 << 3,

    /// <summary>The device wrapper implements <see cref="IParamsFloatSearchIndex"/>.</summary>
    ParamsSearch = 1 << 4,

    /// <summary>The device index is an IVF index, so <see cref="GpuClonerOptions.StorageMode"/> decides what search reports as labels.</summary>
    Ivf = 1 << 5,

    /// <summary>
    /// The device index reports the same distances as the CPU source. Not true for <c>IndexFlat1D</c>: on the CPU it
    /// reports |q - x|, while the device index it becomes is a regular L2 flat reporting (q - x)².
    /// </summary>
    CpuDistances = 1 << 6,
}

public sealed record GpuIndexCase(string Name, int Dimensions, MetricType Metric, Func<int, FloatIndex> CreateCpu, Func<GpuResourcesProvider, FloatIndex, int, FloatIndex> ToGpu, 
    Func<GpuResourcesProvider, FloatIndex, GpuClonerOptions, int, FloatIndex> ToGpuWithOptions, Func<FloatIndex, FloatIndex> ToCpu, GpuCapabilities Capabilities)
{
    /// <summary>Number of training vectors, sized for this index's clustering parameters.</summary>
    public int TrainCount { get; init; }

    /// <summary>
    /// Relative tolerance between device and CPU distances. Float16 device storage also rounds the queries to float16,
    /// so those distances drift by about 1e-3 from a CPU computing in float32.
    /// </summary>
    public float DistanceTolerance { get; init; } = 1e-4f;

    public bool Has(GpuCapabilities caps) => (Capabilities & caps) == caps;

    public FloatIndex CreateUntrainedCpuIndex() => CreateCpu(Dimensions);

    public FloatIndex CreateCpuIndex()
    {
        var index = CreateUntrainedCpuIndex();

        if (Has(GpuCapabilities.Train))
            ((ITrainableFloatIndex)index).TrainAsync(TrainCount, TrainingVectors()).GetAwaiter().GetResult();

        return index;
    }

    public FloatIndex CreatePopulatedCpuIndex(float[] vectors)
    {
        var index = CreateCpuIndex();
        ((IIDSequentialFloatIndex)index).Add(vectors.Length / Dimensions, vectors);
        return index;
    }

    public float[] TrainingVectors() => Vectors.Random(seed: 98, count: TrainCount, dimensions: Dimensions);

    public override string ToString() => Name;
}
