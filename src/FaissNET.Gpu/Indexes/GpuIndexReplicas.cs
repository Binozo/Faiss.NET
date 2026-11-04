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
        get
        {
            var deviceIds = new int[Count];
            for (int i = 0; i < deviceIds.Length; i++)
            {
                deviceIds[i] = base[i].DeviceId;
            }

            return deviceIds;
        }
    }
}