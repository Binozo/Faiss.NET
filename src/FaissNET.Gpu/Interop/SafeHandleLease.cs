using System.Runtime.InteropServices;

namespace Faiss.Gpu.Interop;

internal ref struct SafeHandleLease
{
    private readonly SafeHandle _handle;
    private bool _added;

    public SafeHandleLease(SafeHandle handle)
    {
        _handle = handle;
        handle.DangerousAddRef(ref _added);
    }

    public void Dispose()
    {
        if (_added)
        {
            _added = false;
            _handle.DangerousRelease();
        }
    }
}
