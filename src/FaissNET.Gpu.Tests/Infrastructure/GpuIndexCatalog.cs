using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Models;
using Xunit;

namespace Faiss.Gpu.Tests.Infrastructure;

public static class GpuIndexCatalog
{
    private const GpuCapabilities FlatCaps = GpuCapabilities.Exact | GpuCapabilities.DeviceReconstruct | GpuCapabilities.ParamsSearch | GpuCapabilities.CpuDistances;

    private const GpuCapabilities IvfCaps = GpuCapabilities.Train | GpuCapabilities.DeviceAdd | GpuCapabilities.ParamsSearch | GpuCapabilities.Ivf | GpuCapabilities.CpuDistances;

    private static readonly GpuIndexCase[] Cases =
    [
        Case<IndexFlatL2, GpuIndexFlatL2, IndexFlatL2>("FlatL2", 8, MetricType.L2, d => new IndexFlatL2(d), FlatCaps),
        Case<IndexFlatIP, GpuIndexFlatIP, IndexFlatIP>("FlatIP", 8, MetricType.InnerProduct, d => new IndexFlatIP(d), FlatCaps),
        Case<IndexFlat, GpuIndexFlat, IndexFlat>("FlatGenericL2", 8, MetricType.L2, d => new IndexFlat(d, MetricType.L2), FlatCaps),
        Case<IndexFlat, GpuIndexFlat, IndexFlat>("FlatGenericIP", 8, MetricType.InnerProduct, d => new IndexFlat(d, MetricType.InnerProduct), FlatCaps),

        Case<IndexScalarQuantizer, GpuIndexFlat, IndexFlat>("SQfp16", 8, MetricType.L2, d => new IndexScalarQuantizer(d, QuantizerType.QT_fp16),
            GpuCapabilities.ParamsSearch | GpuCapabilities.CpuDistances) with { DistanceTolerance = 2e-3f },

        Case<IndexIVFFlat<IndexFlatL2>, GpuIndexIVFFlat<IndexFlatL2>, IndexIVFFlat<IndexFlatL2>>("IVFFlatL2", 8, MetricType.L2,
            d => Exhaustive(new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(d), d, nlist: 4)), IvfCaps | GpuCapabilities.Exact, trainCount: 256),
        Case<IndexIVFFlat<IndexFlatIP>, GpuIndexIVFFlat<IndexFlatIP>, IndexIVFFlat<IndexFlatIP>>("IVFFlatIP", 8, MetricType.InnerProduct,
            d => Exhaustive(new IndexIVFFlat<IndexFlatIP>(new IndexFlatIP(d), d, nlist: 4, MetricType.InnerProduct)), IvfCaps | GpuCapabilities.Exact, trainCount: 256),

        Case<IndexIVFScalarQuantizer<IndexFlatL2>, GpuIndexIVFScalarQuantizer<IndexFlatL2>, IndexIVFScalarQuantizer<IndexFlatL2>>("IVFSQ8", 8, MetricType.L2,
            d => Exhaustive(new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(d), d, nlist: 4, QuantizerType.QT_8bit)), IvfCaps, trainCount: 256),
        Case<IndexIVFScalarQuantizer<IndexFlatL2>, GpuIndexIVFScalarQuantizer<IndexFlatL2>, IndexIVFScalarQuantizer<IndexFlatL2>>("IVFSQfp16", 8, MetricType.L2,
            d => Exhaustive(new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(d), d, nlist: 4, QuantizerType.QT_fp16)), IvfCaps, trainCount: 256),

        // PQ4 over d=16 gives 4 sub-quantizers of 4 dimensions; 8-bit codes need >= 256 training points per sub-quantizer
        Case<IndexIVFPQ, GpuIndexIVFPQ, IndexIVFPQ>("IVFPQ", 16, MetricType.L2,
            d => Exhaustive(new IndexIVFPQ(d, n: 4, m: 4)), IvfCaps, trainCount: 1024),
    ];

    public static TheoryData<GpuIndexCase> All { get; } = Where(_ => true);

    public static TheoryData<GpuIndexCase> Exact { get; } = Where(c => c.Has(GpuCapabilities.Exact));

    public static TheoryData<GpuIndexCase> CpuDistanceParity { get; } = Where(c => c.Has(GpuCapabilities.CpuDistances));

    public static TheoryData<GpuIndexCase> DeviceAddable { get; } = Where(c => c.Has(GpuCapabilities.DeviceAdd));

    public static TheoryData<GpuIndexCase> DeviceReconstructable { get; } = Where(c => c.Has(GpuCapabilities.DeviceReconstruct));

    public static TheoryData<GpuIndexCase> ParamsSearchable { get; } = Where(c => c.Has(GpuCapabilities.ParamsSearch));

    public static TheoryData<GpuIndexCase> Ivf { get; } = Where(c => c.Has(GpuCapabilities.Ivf));

    public static GpuIndexCase Get(string name) => Cases.Single(c => c.Name == name);

    public static IEnumerable<GpuIndexCase> With(GpuCapabilities caps) => Cases.Where(c => c.Has(caps));

    private static GpuIndexCase Case<TCpu, TGpu, TBack>(string name, int dimensions, MetricType metric, Func<int, TCpu> create, GpuCapabilities caps, int trainCount = 0)
        where TCpu : FloatIndex, IGpuClonableIndex<TCpu, TGpu>, IFromNativeIndexHandle<TCpu>
        where TGpu : FloatIndex, IGpuIndex<TBack>, IFromNativeIndexHandle<TGpu>
        where TBack : FloatIndex, IFromNativeIndexHandle<TBack>
        => new(name, dimensions, metric,
            d => create(d),
            (provider, cpu, device) => GpuIndexProvider.TransferToGpu(provider, (TCpu)cpu, device),
            (provider, cpu, options, device) => GpuIndexProvider.TransferToGpu(provider, (TCpu)cpu, options, device),
            gpu => GpuIndexProvider.TransferToCpu((TGpu)gpu),
            caps)
        {
            TrainCount = trainCount,
        };

    private static T Exhaustive<T>(T index) where T : IIVFIndex
    {
        index.Nprobe = index.Nlist;
        return index;
    }

    private static TheoryData<GpuIndexCase> Where(Func<GpuIndexCase, bool> predicate)
    {
        var data = new TheoryData<GpuIndexCase>();
        foreach (var c in Cases.Where(predicate))
        {
            data.Add(c);
        }

        return data;
    }
}
