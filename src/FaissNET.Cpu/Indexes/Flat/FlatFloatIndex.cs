using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Flat;

public abstract class FlatFloatIndex<T> : FloatIndex<T>, IRangeSearchFlatIndex where T : FloatIndex<T>
{
    protected FlatFloatIndex(FaissIndexHandle handle) : base(handle)
    {
    }

    protected FlatFloatIndex(IntPtr handle) : base(handle)
    {
    }
    
    
}