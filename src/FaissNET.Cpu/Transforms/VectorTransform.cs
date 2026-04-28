using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Transforms;

public abstract class VectorTransform : IDisposable
{
    internal FaissVectorTransformHandle SafeHandle { get; }

    protected VectorTransform(IntPtr handle)
    {
        SafeHandle = new FaissVectorTransformHandle(handle);
    }
    
    public bool IsTrained => Native.faiss_VectorTransform_is_trained(SafeHandle) != 0;
    
    public int DIn => Native.faiss_VectorTransform_d_in(SafeHandle);
    
    public int DOut => Native.faiss_VectorTransform_d_out(SafeHandle);

    public unsafe void Train(long n, ReadOnlySpan<float> vectors)
    {
        if (!IsTrained)
        {
            fixed (float* pVectors = vectors)
            {
                FaissErrorHandler.ThrowIfError(
                    Native.faiss_VectorTransform_train(SafeHandle, n, pVectors)
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
            Native.faiss_VectorTransform_apply_noalloc(SafeHandle, n, pVectors, pResult);
        }

        return result;
    }

    public unsafe void ReverseTransform(ReadOnlySpan<float> input, Span<float> output)
    {
        fixed (float* pInput = input)
        fixed (float* pOutput = output)
        {
            Native.faiss_VectorTransform_reverse_transform(SafeHandle, input.Length / DOut, pInput, pOutput);
        }
    }

    public void Dispose()
    {
        SafeHandle.Dispose();
        GC.SuppressFinalize(this);
    }

    internal void ReleaseOwnership() => SafeHandle.SetHandleAsInvalid();
}