using Faiss.Cpu.Indexes;
using Faiss.Cpu.Indexes.Replicas;
using Faiss.Cpu.Interfaces;
using Faiss.Interop.SafeHandles;

namespace Faiss.Gpu.Indexes;

public class GpuIndexReplicas<TGpu, TCpu> : IndexReplicas<TGpu>, IGpuDevicesSquadIndex<TCpu>
    where TCpu : INativeIndex, IFromNativeIndexHandle<TCpu>
    where TGpu : FloatIndex, INativeIndex, IFromNativeIndexHandle<TGpu>, IGpuDeviceIndex<TCpu>
{
    private GpuIndexReplicas(long dimensions) : base(dimensions, true, true)
    {
        // Not callable
    }

    private GpuIndexReplicas(FaissIndexHandle handle) : base(handle)
    {
    }

    internal static GpuIndexReplicas<TGpu, TCpu> FromPointer(IntPtr handle) => new(new FaissIndexHandle<IndexReplicasRelease>(handle, ownsHandle: true));
    
    public int[] DeviceIds {
    /// <summary>The device of every replica, in replica order.</summary>
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
                    deviceIds[i] = GpuNative.faiss_GpuIndex_getDevice_ptr(Native.faiss_IndexReplicas_at(NativeHandle, i));

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