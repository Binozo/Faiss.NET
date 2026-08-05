using Faiss.Cpu.Interfaces;
using Faiss.Cpu.Search.Range;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;
using Faiss.Search;

namespace Faiss.Cpu.Indexes.Binary;

internal readonly struct IndexLSHRelease : IFaissRelease
{
    public static void Release(IntPtr handle) => Native.faiss_IndexLSH_free(handle);
}

/// <summary>
/// Locality-Sensitive Hashing index.
/// Hashes vectors into compact binary signatures for fast approximate search.
/// </summary>
public sealed class IndexLSH : FloatIndex, IFromNativeIndexHandle<IndexLSH>, IIDSequentialFloatIndex, IIDMappedFloatIndex, ITrainableFloatIndex, IIDRemovableFloatIndex, IReconstructFloatIndex, IRangeSearchFloatIndex, IComputeResidualFloatIndex, ICodeFloatIndex
{
    public IndexLSH(int dimensions, int nbits, bool rotateData = true, bool trainThresholds = false) : this(CreateHandle(dimensions, nbits, rotateData, trainThresholds))
    {
    }

    private IndexLSH(FaissIndexHandle handle) : base(handle)
    {
        
    }
    
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

    /// <inheritdoc />
    public bool IsTrained => ((ITrainableIndex)this).IsTrained;

    /// <inheritdoc />
    public Task TrainAsync(long count, ReadOnlyMemory<float> vectors) => ((ITrainableFloatIndex)this).TrainAsync(count, vectors);

    /// <inheritdoc />
    public void Add(long count, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            throw new FaissUntrainedException();
        }

        ((IIDSequentialFloatIndex)this).Add(count, vectors);
    }
    
    /// <inheritdoc />
    public void RangeSearch(long count, ReadOnlySpan<float> queryVectors, float radius, RangeSearchResult result) => ((IRangeSearchFloatIndex)this).RangeSearch(count, queryVectors, radius, result);
    
    /// <inheritdoc />
    public long RemoveIds(IIDSelector selector) => ((IIDRemovableFloatIndex)this).RemoveIds(selector);
    
    /// <inheritdoc />
    public float[] Reconstruct(long key) => ((IReconstructFloatIndex)this).Reconstruct(key);

    /// <inheritdoc />
    public float[] Reconstruct(long startKey, long count) => ((IReconstructFloatIndex)this).Reconstruct(startKey, count);
    
    /// <inheritdoc />
    public void ComputeResidual(ReadOnlySpan<float> originalVector, Span<float> residualVector, long key) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVector, residualVector, key);
    
    /// <inheritdoc />
    public void ComputeResidual(ReadOnlySpan<float> originalVectors, Span<float> residualVectors, ReadOnlySpan<long> keys) => ((IComputeResidualFloatIndex)this).ComputeResidual(originalVectors, residualVectors, keys);
    
    /// <inheritdoc />
    public long GetStandaloneCodeSize() => ((ICodeFloatIndex)this).GetStandaloneCodeSize();
    
    /// <inheritdoc />
    public void Encode(long count, ReadOnlySpan<float> vectors, Span<byte> outputBytes) => ((ICodeFloatIndex)this).Encode(count, vectors, outputBytes);
    
    /// <inheritdoc />
    public void Decode(long count, ReadOnlySpan<byte> inputBytes, Span<float> outputVectors) => ((ICodeFloatIndex)this).Decode(count, inputBytes, outputVectors);

    /// <summary>Number of bits per binary signature.</summary>
    public int Nbits => Native.faiss_IndexLSH_nbits(NativeHandle);
    
    /// <summary>Size of the binary code in bytes.</summary>
    public int CodeSize => Native.faiss_IndexLSH_code_size(NativeHandle);
    
    /// <summary>Whether random rotation is applied before hashing.</summary>
    public bool RotateData => Native.faiss_IndexLSH_rotate_data(NativeHandle) != 0;
    
    /// <summary>Whether thresholds are trained from data.</summary>
    public bool TrainThresholds => Native.faiss_IndexLSH_train_thresholds(NativeHandle) != 0;
}