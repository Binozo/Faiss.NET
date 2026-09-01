namespace Faiss.Interop.SafeHandles;

internal class FaissScalarQuantizerHandle : FaissHandle
{
    public FaissScalarQuantizerHandle(bool ownsHandle) : base(ownsHandle)
    {
    }

    public FaissScalarQuantizerHandle(IntPtr preexistingHandle, bool ownsHandle = false) : base(preexistingHandle, ownsHandle)
    {
    }

    protected override bool ReleaseHandle() => true;
}