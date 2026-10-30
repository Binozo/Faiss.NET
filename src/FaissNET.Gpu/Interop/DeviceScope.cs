using System.Runtime.InteropServices;
using Faiss.Exceptions;

namespace Faiss.Gpu.Interop;

/// <summary>
/// Makes a device current for the duration of a native call and restores the previous one afterwards, like faiss's
/// own <c>DeviceScope</c>.
/// </summary>
internal unsafe ref struct DeviceScope
{
    private static readonly delegate* unmanaged[Cdecl]<int*, int> GetDevice;
    private static readonly delegate* unmanaged[Cdecl]<int, int> SetDevice;

    private readonly int _previous;
    private bool _active;

    static DeviceScope()
    {
        var library = NativeLibrary.Load("faiss_c", typeof(DeviceScope).Assembly, null);

        if (NativeLibrary.TryGetExport(library, "cudaGetDevice", out var get) && NativeLibrary.TryGetExport(library, "cudaSetDevice", out var set)
            || NativeLibrary.TryGetExport(library, "hipGetDevice", out get) && NativeLibrary.TryGetExport(library, "hipSetDevice", out set))
        {
            GetDevice = (delegate* unmanaged[Cdecl]<int*, int>)get;
            SetDevice = (delegate* unmanaged[Cdecl]<int, int>)set;
        }
    }

    /// <exception cref="FaissRuntimeException">The GPU runtime could not be resolved or refused to switch devices.</exception>
    public DeviceScope(int device)
    {
        if (GetDevice == null)
            throw new FaissRuntimeException("Cannot switch GPU devices: Not supported in this build.");

        int previous;
        ThrowIfFailed(GetDevice(&previous), "get");
        _previous = previous;

        if (previous != device)
        {
            ThrowIfFailed(SetDevice(device), "set");
            _active = true;
        }
    }

    public void Dispose()
    {
        if (_active)
        {
            _active = false;
            SetDevice(_previous);
        }
    }

    private static void ThrowIfFailed(int error, string operation)
    {
        if (error != 0)
            throw new FaissRuntimeException($"Failed to {operation} the current GPU device (runtime error {error}).");
    }
}
