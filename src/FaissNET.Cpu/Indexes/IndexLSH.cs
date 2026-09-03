using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Cpu.Selectors;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes;

internal readonly struct IndexLSHRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexLSH_free(handle);
}

/// <summary>
/// Locality-Sensitive Hashing index.
/// Hashes vectors into compact binary signatures for fast approximate search.
/// </summary>
public sealed class IndexLSH : FloatIndex, IIDSequentialFloatIndex, ITrainableFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IRangeSearchFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex, ISerializableFloatIndex, IClonableFloatIndex<IndexLSH>, IFromNativeIndexHandle<IndexLSH>
{
    public IndexLSH(int dimensions, int nbits, bool rotateData = true, bool trainThresholds = false) : this(CreateHandle(dimensions, nbits, rotateData, trainThresholds))
    {
    }

    private IndexLSH(FaissIndexHandle handle) : base(handle)
    {
    }

    public bool IsTrained => TrainableFloatIndexImpl.IsTrained(this);

    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => TrainableFloatIndexImpl.TrainAsync(this, count, vectors);

    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        IDSequentialFloatIndexImpl.Add(this, count, vectors);
    }
    
    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => RangeSearchFloatIndexImpl.RangeSearch(this, count, queryVectors, radius, result);
    
    public long RemoveIds(IDSelector selector) => IDRemovableFloatIndexImpl.RemoveIds(this, selector);
    
    public float[] Reconstruct(long key) => ReconstructFloatIndexImpl.Reconstruct(this, key);

    public float[] Reconstruct(long startKey, long count) => ReconstructFloatIndexImpl.Reconstruct(this, startKey, count);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVector, residualVector, key);
    
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ComputeResidualFloatIndexImpl.ComputeResidual(this, originalVectors, residualVectors, keys);
    
    public long GetStandaloneCodeSize() => CodeFloatIndexImpl.GetStandaloneCodeSize(this);
    
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => CodeFloatIndexImpl.Encode(this, count, vectors, outputBytes);
    
    /// <inheritdoc />
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => CodeFloatIndexImpl.Decode(this, count, inputBytes, outputVectors);

    /// <summary>Number of bits per binary signature.</summary>
    public int Nbits => Native.faiss_IndexLSH_nbits(NativeHandle);
    
    /// <summary>Size of the binary code in bytes.</summary>
    public int CodeSize => Native.faiss_IndexLSH_code_size(NativeHandle);
    
    /// <summary>Whether random rotation is applied before hashing.</summary>
    public bool RotateData => Native.faiss_IndexLSH_rotate_data(NativeHandle) != 0;
    
    /// <summary>Whether thresholds are trained from data.</summary>
    public bool TrainThresholds => Native.faiss_IndexLSH_train_thresholds(NativeHandle) != 0;
    
    private static FaissIndexHandle CreateHandle(int dimensions, int nbits, bool rotateData = true, bool trainThresholds = false)
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_IndexLSH_new_with_options(out IntPtr ptr, dimensions, nbits, rotateData, trainThresholds));
        return new FaissIndexHandle<IndexLSHRelease>(ptr);
    }

    private static FaissIndexHandle Wrap(IntPtr handle, bool ownsHandle = true)
        => new FaissIndexHandle<IndexLSHRelease>(handle, ownsHandle);

    static IndexLSH IFromNativeIndexHandle<IndexLSH>.FromPointer(IntPtr handle, bool ownsHandle)
        => new(Wrap(handle, ownsHandle));

    static IndexLSH IFromNativeIndexHandle<IndexLSH>.FromHandle(FaissIndexHandle handle) => new(handle);

    public IndexLSH Clone() => ClonableFloatIndexImpl<IndexLSH>.Clone(this);
}