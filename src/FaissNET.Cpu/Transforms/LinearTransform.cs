using System.Runtime.InteropServices;
using Faiss.Exceptions;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Transforms;

/// <summary>
/// Generic linear map y = A·x + b with a user-supplied matrix A (d_out × d_in, row-major) and optional bias b (d_out).
/// The transform is considered trained as soon as <see cref="SetMatrix"/> (and <see cref="SetBias"/> when <see cref="HaveBias"/> is set) have been called.
/// <see cref="VectorTransform.ReverseTransform"/> is available if <see cref="IsOrthonormal"/>.
/// </summary>
public sealed class LinearTransform : VectorTransform
{
    public LinearTransform(int dIn, int dOut, bool haveBias = false)
        : base(CreateHandle(dIn, dOut, haveBias))
    {
    }

    private LinearTransform(FaissVectorTransformHandle handle) : base(handle)
    {
    }

    internal override VectorTransform CreateView(SafeHandle owner) => new LinearTransform(BorrowFor(owner));

    public override bool IsReversible => IsOrthonormal;

    private static IntPtr CreateHandle(int dIn, int dOut, bool haveBias)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(dIn);
        ArgumentOutOfRangeException.ThrowIfNegative(dOut);

        FaissErrorHandler.ThrowIfError(
            Native.faiss_LinearTransform_new_with(out IntPtr ptr, dIn, dOut, haveBias)
        );

        return ptr;
    }

    /// <summary>
    /// Whether the bias b is added on output. Turning it on requires a bias (<see cref="SetBias"/>) before the transform is trained.
    /// </summary>
    public bool HaveBias
    {
        get => Native.faiss_LinearTransform_have_bias(Handle) != 0;
        set
        {
            Native.faiss_LinearTransform_set_have_bias(Handle, value);
            UpdateIsTrained();
        }
    }

    /// <summary>Whether A has orthonormal rows (AᵀA = I within 4e-5).</summary>
    public bool IsOrthonormal
    {
        get => Native.faiss_LinearTransform_is_orthonormal(Handle) != 0;
        set => Native.faiss_LinearTransform_set_is_orthonormal_flag(Handle, value);
    }

    /// <summary>
    /// Recomputes <see cref="IsOrthonormal"/> from the current matrix (AᵀA = I within 4e-5) and returns it.
    /// Always false when d_out &gt; d_in.
    /// Always true when d_out is 0.
    /// </summary>
    /// <exception cref="InvalidOperationException">The matrix has not been set (see <see cref="SetMatrix"/>).</exception>
    public bool ComputeIsOrthonormal()
    {
        Native.faiss_LinearTransform_A(Handle, out _, out UIntPtr size);

        if (DOut > 0 && DOut <= DIn && size < (ulong)DOut * (ulong)DIn)
            throw new InvalidOperationException("Set the matrix (SetMatrix) before computing orthonormality.");

        Native.faiss_LinearTransform_set_is_orthonormal(Handle);
        return IsOrthonormal;
    }

    /// <summary>A copy of the matrix A (d_out × d_in, row-major), or an empty array if it has not been set.</summary>
    public float[] GetMatrix()
    {
        Native.faiss_LinearTransform_A(Handle, out IntPtr ptr, out UIntPtr size);
        return Copy(ptr, size);
    }

    /// <summary>
    /// Replaces the matrix A (d_out × d_in, row-major) and recomputes <see cref="IsOrthonormal"/>.
    /// </summary>
    /// <exception cref="ArgumentException"><paramref name="matrix"/> does not hold exactly d_out × d_in values.</exception>
    public unsafe void SetMatrix(ReadOnlySpan<float> matrix)
    {
        long expected = (long)DOut * DIn;
        if (matrix.Length != expected)
            throw new ArgumentException($"Matrix must hold d_out × d_in = {expected} values, got {matrix.Length}.", nameof(matrix));

        fixed (float* pMatrix = matrix)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_LinearTransform_set_A(Handle, pMatrix));
        }

        ComputeIsOrthonormal();
        UpdateIsTrained();
    }

    /// <summary>A copy of the bias b (d_out), or an empty array if it has not been set.</summary>
    public float[] GetBias()
    {
        Native.faiss_LinearTransform_b(Handle, out IntPtr ptr, out UIntPtr size);
        return Copy(ptr, size);
    }

    /// <summary>Replaces the bias b (d_out). It is only applied while <see cref="HaveBias"/> is set.</summary>
    /// <exception cref="ArgumentException"><paramref name="bias"/> does not hold exactly d_out values.</exception>
    public unsafe void SetBias(ReadOnlySpan<float> bias)
    {
        if (bias.Length != DOut)
            throw new ArgumentException($"Bias must hold d_out = {DOut} values, got {bias.Length}.", nameof(bias));

        fixed (float* pBias = bias)
        {
            FaissErrorHandler.ThrowIfError(Native.faiss_LinearTransform_set_b(Handle, pBias));
        }

        UpdateIsTrained();
    }

    /// <summary>
    /// Computes x = Aᵀ·(y − b) for a batch of d_out-dimensional <paramref name="input"/> vectors, regardless of
    /// <see cref="IsOrthonormal"/>. Equals <see cref="VectorTransform.ReverseTransform"/> when A has orthonormal rows.
    /// </summary>
    public unsafe void TransformTranspose(ReadOnlySpan<float> input, Span<float> output)
    {
        if (!IsTrained)
            throw new FaissUntrainedException("LinearTransform needs its matrix (and bias, if enabled) set first.");

        if (input.Length == 0 || input.Length % DOut != 0)
            throw new ArgumentException($"Input span length ({input.Length}) must be a positive multiple of d_out ({DOut}).", nameof(input));

        long n = input.Length / DOut;
        long expected = checked(n * DIn);
        if (output.Length < expected)
            throw new ArgumentException($"Output span too small. Expected {expected}, got {output.Length}.", nameof(output));

        fixed (float* pInput = input)
        fixed (float* pOutput = output)
        {
            Native.faiss_LinearTransform_transform_transpose(Handle, n, pInput, pOutput);
        }
    }

    private void UpdateIsTrained()
    {
        Native.faiss_LinearTransform_A(Handle, out _, out UIntPtr matrixSize);
        Native.faiss_LinearTransform_b(Handle, out _, out UIntPtr biasSize);

        bool ready = matrixSize == (ulong)DOut * (ulong)DIn
                     && (!HaveBias || biasSize == (ulong)DOut);

        Native.faiss_VectorTransform_set_is_trained(Handle, ready);
    }

    public bool Verbose
    {
        get => Native.faiss_LinearTransform_verbose(Handle) != 0;
        set => Native.faiss_LinearTransform_set_verbose(Handle, value);
    }

    private static unsafe float[] Copy(IntPtr ptr, UIntPtr size)
    {
        int count = checked((int)size);
        if (count == 0)
            return [];

        return new ReadOnlySpan<float>((void*)ptr, count).ToArray();
    }
}
