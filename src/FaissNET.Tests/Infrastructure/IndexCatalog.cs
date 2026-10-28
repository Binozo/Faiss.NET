using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Indexes.Refine;
using Faiss.Cpu.Serializer;
using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Binary;
using Faiss.Cpu.Indexes.Approximate;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Indexes.Mapped;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Models;
using Xunit;

namespace Faiss.Tests.Infrastructure;

public static class IndexCatalog
{
    private static readonly IndexCase[] FloatCases =
    [
        new("FlatL2", 4, MetricType.L2, d => new IndexFlatL2(d), AllExact)
        {
            CloneIndex = i => ((IndexFlatL2)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexFlatL2>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexFlatL2>(p),
        },

        new("FlatIP", 4, MetricType.InnerProduct, d => new IndexFlatIP(d), AllExact)
        {
            CloneIndex = i => ((IndexFlatIP)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexFlatIP>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexFlatIP>(p),
        },

        // Probing every list makes IVF exhaustive, thus it allows exact-search contracts. Also, the storage is unquantized so reconstruction is bit-exact.
        new("IVFFlatL2", 4, MetricType.L2, d => Exhaustive(new IndexIVFFlat<IndexFlatL2>(new IndexFlatL2(d), d, nlist: 2)),
            IndexCapabilities.SequentialAdd | IndexCapabilities.AddWithIds | IndexCapabilities.Train | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.Reconstruct | IndexCapabilities.Codes |
            IndexCapabilities.Exact | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds)
        {
            TrainCount = 80,
            CloneIndex = i => ((IndexIVFFlat<IndexFlatL2>)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexIVFFlat<IndexFlatL2>>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexIVFFlat<IndexFlatL2>>(p),
            CreateParameters = selector => new SearchParametersIVF(nprobe: 2, maxCodes: 0, selector),
            EnableReconstruct = i => ((IndexIVFFlat<IndexFlatL2>)i).DirectMap = true,
        },

        new("IDMapFlatL2", 4, MetricType.L2, d => new IndexIDMap<IndexFlatL2>(new IndexFlatL2(d)), IndexCapabilities.AddWithIds | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds),

        new("IDMap2FlatL2", 4, MetricType.L2, d => new IndexIDMap2<IndexFlatL2>(new IndexFlatL2(d)), IndexCapabilities.AddWithIds | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds),

        new("HNSWFlatL2", 4, MetricType.L2, d => new IndexHNSWFlat(d, m: 16), HNSW)
        {
            CloneIndex = i => ((IndexHNSWFlat)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexHNSWFlat>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexHNSWFlat>(p),
        },

        new("Shards2FlatL2", 4, MetricType.L2, Sharded, IndexCapabilities.SequentialAdd | IndexCapabilities.Exact | IndexCapabilities.ParamsSearch),

        new("PQ", 8, MetricType.L2, d => new IndexPQ(d, m: 4), Lossy | IndexCapabilities.RemoveIds | IndexCapabilities.LossyCodes)
        {
            TrainCount = 512,
            CloneIndex = i => ((IndexPQ)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexPQ>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexPQ>(p),
        },

        new("LSH", 8, MetricType.L2, d => new IndexLSH(d, nbits: 32),
            IndexCapabilities.SequentialAdd | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.LossyReconstruct | IndexCapabilities.RemoveIds |
            IndexCapabilities.LossyCodes)
        {
            CloneIndex = i => ((IndexLSH)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexLSH>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexLSH>(p),
        },

        new("HNSWPQ", 8, MetricType.L2, d => new IndexHNSWPQ(d, m: 16, productQuantization: 4), Lossy | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch)
        {
            TrainCount = 512,
            CloneIndex = i => ((IndexHNSWPQ)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexHNSWPQ>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexHNSWPQ>(p),
        },

        new("HNSWSQ", 8, MetricType.L2, d => new IndexHNSWSQ(d, QuantizerType.QT_8bit, m: 16), Lossy | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch)
        {
            TrainCount = 512,
            CloneIndex = i => ((IndexHNSWSQ)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexHNSWSQ>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexHNSWSQ>(p),
        },

        new("HNSWFlatPanorama", 8, MetricType.L2, d => new IndexHNSWFlatPanorama(d, m: 16),
            IndexCapabilities.SequentialAdd | IndexCapabilities.Clone | IndexCapabilities.Reconstruct | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch)
        {
            CloneIndex = i => ((IndexHNSWFlatPanorama)i).Clone(),
        },

        // NSG must get more than 100 vectors (NNDescent.cpp:429) which is not handled in this catalogue, thus they get tested in dedicated tests.
        // Capability none will still test their construction/lifetime sets.
        new("NSGFlat", 8, MetricType.L2, d => new IndexNSGFlat(d, r: 16), IndexCapabilities.None),
        new("NSGPQ", 8, MetricType.L2, d => new IndexNSGPQ(d, r: 16, productQuantization: 4), IndexCapabilities.None),
        new("NSGSQ", 8, MetricType.L2, d => new IndexNSGSQ(d, r: 16), IndexCapabilities.None),

        new("FlatGenericL2", 4, MetricType.L2, d => new IndexFlat(d, MetricType.L2), AllExact)
        {
            CloneIndex = i => ((IndexFlat)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexFlat>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexFlat>(p),
        },

        new("FlatGenericIP", 4, MetricType.InnerProduct, d => new IndexFlat(d, MetricType.InnerProduct), AllExact)
        {
            CloneIndex = i => ((IndexFlat)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexFlat>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexFlat>(p),
        },

        new("IVFPQ", 8, MetricType.L2, d => Exhaustive(new IndexIVFPQ(d, n: 2, m: 4)),
            Lossy | IndexCapabilities.AddWithIds | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch |
            IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds | IndexCapabilities.LossyCodes)
        {
            TrainCount = 512,
            CloneIndex = i => ((IndexIVFPQ)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexIVFPQ>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexIVFPQ>(p),
            CreateParameters = selector => new SearchParametersIVF(nprobe: 2, maxCodes: 0, selector),
            EnableReconstruct = i => ((IndexIVFPQ)i).DirectMap = true,
        },

        new("IVFSQ8", 8, MetricType.L2,
            d => ExhaustiveSq(new IndexIVFScalarQuantizer<IndexFlatL2>(new IndexFlatL2(d), d, nlist: 2, QuantizerType.QT_8bit, ownQuantizer: true)),
            Lossy | IndexCapabilities.AddWithIds | IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch |
            IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds | IndexCapabilities.LossyCodes)
        {
            TrainCount = 256,
            CloneIndex = i => ((IndexIVFScalarQuantizer<IndexFlatL2>)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexIVFScalarQuantizer<IndexFlatL2>>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexIVFScalarQuantizer<IndexFlatL2>>(p),
            CreateParameters = selector => new SearchParametersIVF(nprobe: 2, maxCodes: 0, selector),
            EnableReconstruct = i => ((IndexIVFScalarQuantizer<IndexFlatL2>)i).DirectMap = true,
        },

        // TODO: Enable clone and deserialization as soon as PR #5514 https://github.com/facebookresearch/faiss/pull/5514 gets merged
        new("RefineFlatL2", 4, MetricType.L2, d => new IndexRefineFlat<IndexFlatL2>(new IndexFlatL2(d)),
            IndexCapabilities.SequentialAdd | IndexCapabilities.Exact | IndexCapabilities.Reconstruct | IndexCapabilities.Codes | IndexCapabilities.RangeSearch),

        new("SQ8", 8, MetricType.L2, d => new IndexScalarQuantizer(d, QuantizerType.QT_8bit),
            IndexCapabilities.SequentialAdd | IndexCapabilities.Train | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.SelectorSearch |
            IndexCapabilities.RemoveIds | IndexCapabilities.LossyCodes | IndexCapabilities.NearestReconstruction)
        {
            TrainCount = 128,
            CloneIndex = i => ((IndexScalarQuantizer)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<IndexScalarQuantizer>(s),
            DeserializeFile = p => IndexDeserializer.Read<IndexScalarQuantizer>(p),
        },

        new("FactoryFlatL2", 4, MetricType.L2, d => IndexFactory.Create<GenericFloatIndex>("Flat", d, MetricType.L2), AllExact)
        {
            CloneIndex = i => ((GenericFloatIndex)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<GenericFloatIndex>(s),
            DeserializeFile = p => IndexDeserializer.Read<GenericFloatIndex>(p),
        },

        new("FactoryFlatIP", 4, MetricType.InnerProduct, d => IndexFactory.Create<GenericFloatIndex>("Flat", d, MetricType.InnerProduct), AllExact)
        {
            CloneIndex = i => ((GenericFloatIndex)i).Clone(),
            Deserialize = s => IndexDeserializer.Read<GenericFloatIndex>(s),
            DeserializeFile = p => IndexDeserializer.Read<GenericFloatIndex>(p),
        },

        new("FactoryIDMapFlatL2", 4, MetricType.L2, d => IndexFactory.Create<GenericFloatIndex>("IDMap,Flat", d, MetricType.L2),
            IndexCapabilities.AddWithIds | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds),
    ];

    private static readonly IndexCase[] PendingNativeSupport =
    [
        // TODO: Blocked until PR #5511 https://github.com/facebookresearch/faiss/pull/5511 gets merged
        new("ReplicasFlatL2", 4, MetricType.L2, Replicated, IndexCapabilities.SequentialAdd | IndexCapabilities.Exact | IndexCapabilities.Reconstruct),
    ];

    public static TheoryData<IndexCase> PendingFloat { get; } = ToTheory(PendingNativeSupport);

    private static IndexShards Sharded(int dimensions)
    {
        var shards = new IndexShards(dimensions);
        shards.AddIndex(new IndexFlatL2(dimensions));
        shards.AddIndex(new IndexFlatL2(dimensions));
        return shards;
    }

    private static IndexIVFFlat<IndexFlatL2> Exhaustive(IndexIVFFlat<IndexFlatL2> index)
    {
        index.Nprobe = index.Nlist;
        return index;
    }

    private static IndexIVFPQ Exhaustive(IndexIVFPQ index)
    {
        index.Nprobe = index.Nlist;
        return index;
    }

    private static IndexIVFScalarQuantizer<IndexFlatL2> ExhaustiveSq(IndexIVFScalarQuantizer<IndexFlatL2> index)
    {
        index.Nprobe = index.Nlist;
        return index;
    }

    private static IndexReplicas<IndexFlatL2> Replicated(int dimensions)
    {
        var replicas = new IndexReplicas<IndexFlatL2>(dimensions);
        replicas.AddReplica(new IndexFlatL2(dimensions));
        replicas.AddReplica(new IndexFlatL2(dimensions));
        return replicas;
    }

    private static readonly BinaryIndexCase[] BinaryCases =
    [
        // faiss IndexBinary carries METRIC_L2 in its metric_type field even though distances are Hamming.
        new("BinaryFlat", 16, MetricType.L2, d => new IndexBinaryFlat(d), AllExact)
        {
            CloneIndex = i => ((IndexBinaryFlat)i).Clone(),
            Deserialize = s => BinaryIndexDeserializer.Read<IndexBinaryFlat>(s),
            DeserializeFile = p => BinaryIndexDeserializer.Read<IndexBinaryFlat>(p),
        },

        new("BinaryIVF", 16, MetricType.L2, d => Exhaustive(new IndexBinaryIVF(d, nlist: 2, hnswM: null)),
            IndexCapabilities.SequentialAdd | IndexCapabilities.AddWithIds | IndexCapabilities.Train | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.Exact | IndexCapabilities.ParamsSearch |
            IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds)
        {
            TrainCount = 64,
            CloneIndex = i => ((IndexBinaryIVF)i).Clone(),
            Deserialize = s => BinaryIndexDeserializer.Read<IndexBinaryIVF>(s),
            DeserializeFile = p => BinaryIndexDeserializer.Read<IndexBinaryIVF>(p),
            CreateParameters = selector => new SearchParametersIVF(nprobe: 2, maxCodes: 0, selector),
        },

        new("BinaryIDMapFlat", 16, MetricType.L2, d => new IndexBinaryIDMap<IndexBinaryFlat>("BFlat", d), IndexCapabilities.AddWithIds | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds),

        new("BinaryIDMap2Flat", 16, MetricType.L2, d => new IndexBinaryIDMap2<IndexBinaryFlat>("BFlat", d), IndexCapabilities.AddWithIds | IndexCapabilities.RemoveIds | IndexCapabilities.RemoveKeepsIds),

        new("BinaryHash", 16, MetricType.L2, d => new IndexBinaryHash(d, leadingBits: 4), IndexCapabilities.SequentialAdd | IndexCapabilities.AddWithIds | IndexCapabilities.Serialize)
        {
            Deserialize = s => BinaryIndexDeserializer.Read<IndexBinaryHash>(s),
            DeserializeFile = p => BinaryIndexDeserializer.Read<IndexBinaryHash>(p),
        },
    ];

    private static IndexBinaryIVF Exhaustive(IndexBinaryIVF index)
    {
        index.NProbe = index.Nlist;
        return index;
    }

    private const IndexCapabilities AllExact =
        IndexCapabilities.SequentialAdd | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.Reconstruct | IndexCapabilities.Codes | IndexCapabilities.Exact |
        IndexCapabilities.SelectorSearch | IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch | IndexCapabilities.RemoveIds;

    private const IndexCapabilities HNSW =
        IndexCapabilities.SequentialAdd | IndexCapabilities.Clone | IndexCapabilities.Serialize | IndexCapabilities.Reconstruct | IndexCapabilities.SelectorSearch |
        IndexCapabilities.RangeSearch | IndexCapabilities.ParamsSearch;

    private const IndexCapabilities Lossy =
        IndexCapabilities.SequentialAdd | IndexCapabilities.Train | IndexCapabilities.Clone | IndexCapabilities.Serialize |
        IndexCapabilities.LossyReconstruct | IndexCapabilities.NearestReconstruction;

    public static TheoryData<IndexCase> Float { get; } = ToTheory(FloatCases);

    public static TheoryData<IndexCase> RemovableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.RemoveIds)));

    public static TheoryData<BinaryIndexCase> RemovableBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.RemoveIds)));

    public static TheoryData<BinaryIndexCase> Binary { get; } = ToTheory(BinaryCases);

    public static TheoryData<IndexCase> ParamsSearchFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.ParamsSearch)));

    public static TheoryData<BinaryIndexCase> ParamsSearchBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.ParamsSearch)));

    public static TheoryData<BinaryIndexCase> ExactBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Exact)));

    public static TheoryData<IndexCase> SequentialFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd)));

    public static TheoryData<BinaryIndexCase> SequentialBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd)));

    public static TheoryData<IndexCase> ClonableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Clone)));

    public static TheoryData<BinaryIndexCase> ClonableBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Clone)));

    public static TheoryData<IndexCase> SerializableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Serialize)));

    public static TheoryData<BinaryIndexCase> SerializableBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Serialize)));

    public static TheoryData<BinaryIndexCase> MappedBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.AddWithIds)));

    public static TheoryData<BinaryIndexCase> SelectorBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.SelectorSearch)));

    public static TheoryData<IndexCase> ReconstructableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Reconstruct)));

    public static TheoryData<IndexCase> LossyReconstructableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.LossyReconstruct)));

    public static TheoryData<IndexCase> NearestLossyReconstructableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.LossyReconstruct | IndexCapabilities.NearestReconstruction)));

    public static TheoryData<BinaryIndexCase> ReconstructableBinary { get; } =
        ToTheory(BinaryCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Reconstruct)));

    public static TheoryData<IndexCase> TrainableFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.Train)));

    public static TheoryData<IndexCase> MappedFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.AddWithIds)));

    public static TheoryData<IndexCase> CodeFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.Codes)));

    public static TheoryData<IndexCase> LossyCodeFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.LossyCodes)));

    public static TheoryData<IndexCase> NearestLossyCodeFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.LossyCodes | IndexCapabilities.NearestReconstruction)));

    public static TheoryData<IndexCase> ExactFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Exact)));

    public static TheoryData<IndexCase> ApproximateFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd) && !c.Has(IndexCapabilities.Exact)));

    public static TheoryData<IndexCase> SelectorFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Exact | IndexCapabilities.SelectorSearch)));

    public static TheoryData<IndexCase> ApproximateSelectorFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.SelectorSearch) && !c.Has(IndexCapabilities.Exact)));

    public static TheoryData<IndexCase> RangeSearchFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.Exact | IndexCapabilities.RangeSearch)));

    public static TheoryData<IndexCase> ApproximateRangeSearchFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.RangeSearch | IndexCapabilities.Reconstruct) && !c.Has(IndexCapabilities.Exact)));

    public static TheoryData<IndexCase> LossyRangeSearchFloat { get; } =
        ToTheory(FloatCases.Where(c => c.Has(IndexCapabilities.SequentialAdd | IndexCapabilities.RangeSearch | IndexCapabilities.LossyReconstruct)));

    private static TheoryData<T> ToTheory<T>(IEnumerable<T> cases) where T : class
    {
        var data = new TheoryData<T>();
        foreach (var item in cases)
        {
            data.Add(item);
        }

        return data;
    }
}
