using System.Runtime.InteropServices;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.Errors;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.AutoTune;

/// <summary>
/// Represents a parameter space for auto-tuning index parameters.
/// </summary>
public sealed class ParameterSpace : IDisposable
{
    private readonly FaissParameterSpaceHandle _handle;

    public ParameterSpace()
    {
        FaissErrorHandler.ThrowIfError(Native.faiss_ParameterSpace_new(out IntPtr ptr));

        _handle = new FaissParameterSpaceHandle(ptr);
    }

    /// <summary>
    /// Gets the total number of parameter combinations (product of range sizes).
    /// </summary>
    public int CombinationCount => (int)Native.faiss_ParameterSpace_n_combinations(_handle);

    /// <summary>
    /// Gets the string representation of a combination by index.
    /// </summary>
    /// <param name="cno">The combination index.</param>
    /// <returns>The name of the combination.</returns>
    public unsafe string GetCombinationName(int cno)
    {
        const int bufSize = 1000;
        byte* buf = stackalloc byte[bufSize];

        FaissErrorHandler.ThrowIfError(
            Native.faiss_ParameterSpace_combination_name(_handle, (UIntPtr)cno, buf, bufSize)
        );

        return Marshal.PtrToStringAnsi((IntPtr)buf)!;
    }

    /// <summary>
    /// Applies a parameter combination described by a string.
    /// </summary>
    /// <param name="index">The index to configure.</param>
    /// <param name="description">The parameter combination description (e.g., "nprobe=16,efSearch=128").</param>
    public void SetParameters(INativeIndex index, string description)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_ParameterSpace_set_index_parameters(_handle, index.Handle, description)
        );
    }

    /// <summary>
    /// Applies a parameter combination by index number.
    /// </summary>
    /// <param name="index">The index to configure.</param>
    /// <param name="combinationNo">The combination index.</param>
    public void SetParameters(INativeIndex index, int combinationNo)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_ParameterSpace_set_index_parameters_cno(_handle, index.Handle, (UIntPtr)combinationNo)
        );
    }

    /// <summary>
    /// Sets a single parameter by name.
    /// </summary>
    /// <param name="index">The index to configure.</param>
    /// <param name="name">The name of the parameter.</param>
    /// <param name="value">The value to set.</param>
    public void SetParameter(INativeIndex index, string name, double value)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_ParameterSpace_set_index_parameter(_handle, index.Handle, name, value)
        );
    }

    /// <summary>
    /// Displays the parameter space description.
    /// </summary>
    public void Display() => Native.faiss_ParameterSpace_display(_handle);
    
    /// <summary>
    /// Adds a new parameter range with explicit values.
    /// </summary>
    /// <param name="name">The name of the parameter.</param>
    /// <param name="values">The possible values for the parameter.</param>
    /// <returns>The added parameter range.</returns>
    public unsafe ParameterRange AddRange(string name, ReadOnlySpan<double> values)
    {
        FaissErrorHandler.ThrowIfError(
            Native.faiss_ParameterSpace_add_range(_handle, name, out IntPtr rangePtr)
        );

        return new ParameterRange(rangePtr);
    }

    public void Dispose()
    {
        _handle.Dispose();
        GC.SuppressFinalize(this);
    }
}