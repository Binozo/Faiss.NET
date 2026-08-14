using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Applies a transformation to a batch of float vectors: n×d_in in, n×d_out out.
/// <see cref="Train"/> learns the transform from representative data (no-op for data-independent transforms).
/// <see cref="Apply"/> returns a newly allocated transformed batch.
/// <see cref="ReverseTransform"/> inverts it where supported (possibly approximately).
/// Transforms are chained inside IndexPreTransform to pre-process vectors before indexing.
/// </summary>
public abstract class VectorTransform : IDisposable
{
    internal readonly FaissVectorTransformHandle Handle;

    protected VectorTransform(IntPtr handle)
    {
        Handle = new FaissVectorTransformHandle(handle);
    }
    
    public bool IsTrained => Native.faiss_VectorTransform_is_trained(Handle) != 0;
    
    public int DIn => Native.faiss_VectorTransform_d_in(Handle);
    
    public int DOut => Native.faiss_VectorTransform_d_out(Handle);

    public unsafe void Train(long n, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            fixed (float* pVectors = vectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_VectorTransform_train(Handle, n, pVectors)
                );
            }
        }
    }

    public unsafe float[] Apply(long n, ReadOnlySpan<float> vectors)
    {
        float[] result = new float[n * DOut];
        fixed (float* pVectors = vectors)
        fixed (float* pResult = result)
        {
            Native.faiss_VectorTransform_apply_noalloc(Handle, n, pVectors, pResult);
        }

        return result;
    }

    public unsafe void ReverseTransform(ReadOnlySpan<float> input, Span<float> output)
    {
        fixed (float* pInput = input)
        fixed (float* pOutput = output)
        {
            Native.faiss_VectorTransform_reverse_transform(Handle, input.Length / DOut, pInput, pOutput);
        }
    }

    public void Dispose()
    {
        Handle.Dispose();
        GC.SuppressFinalize(this);
    }

    internal void ReleaseOwnership() => Handle.SetHandleAsInvalid();
}