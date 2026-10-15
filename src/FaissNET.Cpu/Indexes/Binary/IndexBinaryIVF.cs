using Faiss.Cpu.Factory;
using Faiss.Cpu.Indexes.IVF;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Models;

namespace Faiss.Cpu.Indexes.Binary;

internal readonly struct IndexBinaryIVFRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexBinaryIVF_free(handle);
}

/// <summary>
/// Binary inverted file index with coarse quantization and binary flat lists.
/// </summary>
public sealed class IndexBinaryIVF : BinaryIndex, ITrainableBinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IParamsBinarySearchIndex, IRangeSearchBinaryIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex,
    ICpuBinaryIndex, ISerializableBinaryIndex, IClonableBinaryIndex<IndexBinaryIVF>, IFromNativeBinaryIndexHandle<IndexBinaryIVF>
{
    /// <summary>
    /// Creates a binary IVF index.
    /// </summary>
    /// <param name="dimensions">Vector dimensionality in bits.</param>
    /// <param name="nlist">Number of inverted lists (coarse clusters).</param>
    /// <param name="hnswM">If provided, this index is created including IndexBinaryHNSW as quantizer</param>
    public IndexBinaryIVF(int dimensions, int nlist, int? hnswM) : this(CreateHandle(dimensions, nlist, hnswM))
    {
    }

    private IndexBinaryIVF(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => TrainableBinaryIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) => TrainableBinaryIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<byte> vectors) => IDSequentialBinaryIndexImpl.Add(this, count, vectors);
    
    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => IDMappedBinaryIndexImpl.Add(this, count, vectors, xids);

    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => RangeSearchBinaryIndexImpl.RangeSearch(this, count, queryVectors, radius, result);
    
    public long RemoveIds(IDSelector selector) => IDRemovableBinaryIndexImpl.RemoveIds(this, selector);

    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParameters parameters, Span<int> distances, Span<long> labels)
    {
        ValidateSearch(count, parameters);
        ParamsBinarySearchIndexImpl.SearchWithParams(this, count, queryVectors, k, parameters, distances, labels);
    }

    /// <inheritdoc cref="IParamsBinarySearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParametersIVF parameters, Span<int> distances, Span<long> labels) =>
        SearchWithParams(count, queryVectors, k, (SearchParameters)parameters, distances, labels);

    public byte[] Reconstruct(long key) => ReconstructBinaryIndexImpl.Reconstruct(this, key);

    public byte[] Reconstruct(long startKey, long count) => ReconstructBinaryIndexImpl.Reconstruct(this, startKey, count);

    public int Nlist => (int)Native.faiss_IndexBinaryIVF_nlist(NativeHandle);

    public int NProbe
    {
        get => (int)Native.faiss_IndexBinaryIVF_nprobe(NativeHandle);
        set => Native.faiss_IndexBinaryIVF_set_nprobe(NativeHandle, (nuint)value);
    }

    public bool OwnQuantizer
    {
        get => Native.faiss_IndexBinaryIVF_own_fields(NativeHandle) != 0;
        private set => Native.faiss_IndexBinaryIVF_set_own_fields(NativeHandle, value);
    }

    public int MaxCodes
    {
        get => (int)Native.faiss_IndexBinaryIVF_max_codes(NativeHandle);
        set => Native.faiss_IndexBinaryIVF_set_max_codes(NativeHandle, (nuint)value);
    }

    public bool UseHeap
    {
        get => Native.faiss_IndexBinaryIVF_use_heap(NativeHandle) != 0;
        set => Native.faiss_IndexBinaryIVF_set_use_heap(NativeHandle, value);
    }

    public bool PerInvlistSearch
    {
        get => Native.faiss_IndexBinaryIVF_per_invlist_search(NativeHandle) != 0;
        set => Native.faiss_IndexBinaryIVF_set_per_invlist_search(NativeHandle, value);
    }

    private protected override void ValidateSearch(long count, SearchParameters? parameters)
    {
        var ivf = parameters as SearchParametersIVF;
        long maxCodes = ivf?.MaxCodes ?? MaxCodes;

        if (PerInvlistSearch)
        {
            if (maxCodes != 0)
                throw new InvalidOperationException("PerInvlistSearch requires a code budget (MaxCodes) of 0.");

            long nlist = Nlist;
            long nprobe = Math.Min(nlist, ivf?.Nprobe ?? NProbe);
            if (count < nlist || count * nprobe <= nlist)
                throw new ArgumentOutOfRangeException(nameof(count), count,
                    $"PerInvlistSearch needs at least {nlist} queries (nlist) and more than {nlist} query × probe pairs; got {count} queries with nprobe {nprobe}.");
        }
        else if (!UseHeap && maxCodes == 0)
        {
            throw new InvalidOperationException("Counting search (UseHeap = false) requires a positive code budget (MaxCodes); with 0, faiss scans nothing.");
        }
    }

    public double ImbalanceFactor => Native.faiss_IndexBinaryIVF_imbalance_factor(NativeHandle);

    /// <inheritdoc cref="IndexIVFFlat{T}.DirectMap"/>
    public void MakeDirectMap(bool maintainDirectMap) => BinaryDirectMapImpl.Make(NativeHandle, maintainDirectMap);

    /// <summary>
    /// Sets the direct map type.
    /// <see cref="Models.DirectMapType.Hashtable"/> supports non-sequential ids.
    /// </summary>
    /// <remarks>
    /// If building the map fails, the index is left without a direct map.
    /// </remarks>
    public DirectMapType DirectMapType
    {
        set => BinaryDirectMapImpl.SetType(NativeHandle, value);
    }

    private static FaissBinaryIndexHandle CreateHandle(int dimensions, int nlist, int? hnswM)
    {
        if (dimensions == 0 || dimensions % 8 != 0)
        {
            throw new ArgumentException("Dimensions must be divisible by 8", nameof(dimensions));
        }

        string description = $"BIVF{nlist}";
        if (hnswM.HasValue)
        {
            description = $"{description}_HNSW{hnswM.Value}";
        }

        return BinaryIndexFactory.Create<IndexBinaryIVF>(description, dimensions).NativeHandle;
    }

    static IndexBinaryIVF IFromNativeBinaryIndexHandle<IndexBinaryIVF>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

    private static FaissBinaryIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissBinaryIndexHandle<IndexBinaryIVFRelease>(handle, ownsHandle);

    static IndexBinaryIVF IFromNativeBinaryIndexHandle<IndexBinaryIVF>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    public IndexBinaryIVF Clone() => ClonableBinaryIndexImpl<IndexBinaryIVF>.Clone(this);
}