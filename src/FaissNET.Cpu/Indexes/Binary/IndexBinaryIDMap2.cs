using Faiss.Cpu.Factory;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Cpu.Indexes.Binary;

public sealed class IndexBinaryIDMap2<T> : MappedBinaryIndex<IndexBinaryIDMap2<T>, T>, IFromNativeBinaryIndexHandle<IndexBinaryIDMap2<T>>, IReconstructBinaryIndex where T : IIDSequentialBinaryIndex, IBinaryIndex, IFromNativeBinaryIndexHandle<T>
{
    public IndexBinaryIDMap2(string description, int dimension) : this(CreateHandle($"IDMap2,{description}", dimension))
    {
    }

    private IndexBinaryIDMap2(FaissBinaryIndexHandle handle) : base(handle)
    {
    }

    private static FaissBinaryIndexHandle CreateHandle(string description, int dimension) => BinaryIndexFactory.Create<IndexBinaryIDMap2<T>>(description, dimension).NativeHandle;

    static IndexBinaryIDMap2<T> IFromNativeBinaryIndexHandle<IndexBinaryIDMap2<T>>.FromHandle(FaissBinaryIndexHandle handle) => new(handle);

    public byte[] Reconstruct(long key) =>  ((IReconstructBinaryIndex)this).Reconstruct(key);

    public byte[] Reconstruct(long startKey, long count)  => ((IReconstructBinaryIndex)this).Reconstruct(startKey, count);
}