using System.Runtime.InteropServices;
using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.AutoTune;

/// <summary>
/// Represents a single tunable parameter and its possible values.
/// </summary>
public sealed class ParameterRange
{
    internal IntPtr Handle { get; }

    internal ParameterRange(IntPtr handle)
    {
        Handle = handle;
    }

    /// <summary>
    /// Gets the name of the parameter.
    /// </summary>
    public unsafe string Name
    {
        get
        {
            IntPtr namePtr = Native.faiss_ParameterRange_name(Handle);
            return Marshal.PtrToStringAnsi(namePtr)!;
        }
    }

    /// <summary>
    /// Gets the possible values for the parameter.
    /// </summary>
    public unsafe ReadOnlySpan<double> Values
    {
        get
        {
            Native.faiss_ParameterRange_values(Handle, out IntPtr valuesPtr, out UIntPtr size);
            return new ReadOnlySpan<double>((double*)valuesPtr, (int)size);
        }
    }
}