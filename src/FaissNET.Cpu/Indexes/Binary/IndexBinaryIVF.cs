using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Parameters;
using Faiss.Cpu.Search.Range;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Search;
using ITrainableBinaryIndex = Faiss.Cpu.Interfaces.ITrainableBinaryIndex;

namespace Faiss.Cpu.Indexes.Binary;

internal readonly struct IndexBinaryIVFRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexBinaryIVF_free(handle);
}

/// <summary>
/// Binary inverted file index with coarse quantization and binary flat lists.
/// </summary>
public sealed class IndexBinaryIVF : BinaryIndex, ITrainableBinaryIndex, IIDSequentialBinaryIndex, IIDMappedBinaryIndex, IParamsBinarySearchIndex, IRangeSearchBinaryIndex, IIDRemovableBinaryIndex, IReconstructBinaryIndex, ISerializableBinaryIndex, IClonableBinaryIndex<IndexBinaryIVF>, IFromNativeBinaryIndexHandle<IndexBinaryIVF>
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
    
    private static FaissBinaryIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
     => new FaissBinaryIndexHandle<IndexBinaryIVFRelease>(handle, ownsHandle);

    static IndexBinaryIVF IFromNativeBinaryIndexHandle<IndexBinaryIVF>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexBinaryIVF IFromNativeBinaryIndexHandle<IndexBinaryIVF>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

    /// <inheritdoc/>
    public bool IsTrained => ((ITrainableBinaryIndex)this).IsTrained;
    
    /// <inheritdoc/>
    public Task TrainAsync(long count, ReadOnlyMemory<byte> vectors) => ((ITrainableBinaryIndex)this).TrainAsync(count, vectors);

    /// <inheritdoc/>
    public void Add(long count, ReadOnlySpan<byte> vectors) => ((IIDSequentialBinaryIndex)this).Add(count, vectors);

    /// <inheritdoc/>
    public void Add(long count, ReadOnlySpan<byte> vectors, ReadOnlySpan<long> xids) => ((IIDMappedBinaryIndex)this).Add(count, vectors, xids);

    /// <inheritdoc cref="IParamsBinarySearchIndex" />
    public void SearchWithParams(long count, ReadOnlySpan<byte> queryVectors, int k, SearchParametersIVF parameters, Span<int> distances, Span<long> labels) =>
        ((IParamsBinarySearchIndex)this).SearchWithParams(count, queryVectors, k, parameters, distances, labels);
    
    /// <inheritdoc/>
    public void RangeSearch(long count, ReadOnlySpan<byte> queryVectors, byte radius, RangeSearchResult result) => ((IRangeSearchBinaryIndex)this).RangeSearch(count, queryVectors, radius, result);

    /// <inheritdoc/>
    public long RemoveIds(IIDSelector selector) => ((IIDRemovableBinaryIndex)this).RemoveIds(selector);

    /// <inheritdoc/>
    public byte[] Reconstruct(long key)
    {
        MakeDirectMap(true);
        return ((IReconstructBinaryIndex)this).Reconstruct(key);
    }

    /// <inheritdoc/>
    public byte[] Reconstruct(long startKey, long count)
    {
        MakeDirectMap(true);
        return ((IReconstructBinaryIndex)this).Reconstruct(startKey, count);
    }

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

    public double ImbalanceFactor =>
        Native.faiss_IndexBinaryIVF_imbalance_factor(NativeHandle);

    public void MakeDirectMap(bool maintainDirectMap) => FaissErrorHandler.ThrowIfError(Native.faiss_IndexBinaryIVF_make_direct_map(NativeHandle, maintainDirectMap));

    public IndexBinaryIVF Clone() => ((IClonableBinaryIndex<IndexBinaryIVF>)this).Clone();
}