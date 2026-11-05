using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Factory;
using Faiss.Cpu.Indexes.Sharding;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.NativeMethods;
using Faiss.Interop.SafeHandles;

namespace Faiss.Gpu.Indexes;

public class GpuIndexShards<TGpu, TCpu> : IndexShards, IGpuDevicesSquadIndex<TCpu>
    where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
    where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
{

    private GpuIndexShards(FaissIndexHandle handle) : base(handle)
    {
    }

    internal static GpuIndexShards<TGpu, TCpu> FromPointer(IntPtr handle) => new(new FaissIndexHandle<IndexShardsRelease>(handle, ownsHandle: true));
    
    public int[] DeviceIds {

    /// <inheritdoc cref="IndexShards.this[int]"/>
    public new TGpu this[int index] => (TGpu)base[index];

    private protected override INativeIndex CreateView(IntPtr member) => TGpu.FromBorrowed(member, NativeHandle);

    public override void AddIndex(INativeIndex index)
    {
        if (index is not TGpu)
            throw new ArgumentException($"Only {typeof(TGpu).Name} shards can be added.", nameof(index));

        base.AddIndex(index);
        QuantizerType = GpuScalarQuantizerLimits.Combine(QuantizerType, index);
    }
    /// <summary>The device of every shard, in shard order.</summary>
    public int[] DeviceIds
    {
        get
        {
            var added = false;
            NativeHandle.DangerousAddRef(ref added);
            try
            {
                var deviceIds = new int[Count];
                for (var i = 0; i < deviceIds.Length; i++)
                    deviceIds[i] = GpuNative.faiss_GpuIndex_getDevice_ptr(Native.faiss_IndexShards_at(NativeHandle, i));

                return deviceIds;
            }
            finally
            {
                if (added)
                    NativeHandle.DangerousRelease();
            }
        }
    }
}