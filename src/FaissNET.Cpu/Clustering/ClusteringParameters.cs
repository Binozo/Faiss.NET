using Faiss.Interop.NativeMethods;

namespace Faiss.Cpu.Clustering;

public sealed class ClusteringOptions
{
    public int Niter { get; set; } = 25;
    public int Nredo { get; set; } = 1;
    public bool Verbose { get; set; }
    public bool Spherical { get; set; }
    public bool IntCentroids { get; set; }
    public bool UpdateIndex { get; set; }
    public bool FrozenCentroids { get; set; }
    public int MinPointsPerCentroid { get; set; }
    public int MaxPointsPerCentroid { get; set; }
    public int Seed { get; set; } = 1234;
    public int DecodeBlockSize { get; set; } = 32768;

    internal Native.ClusteringParameters ToNative()
    {
        var cp = new Native.ClusteringParameters();
        Native.faiss_ClusteringParameters_init(ref cp); // apply C++ defaults

        cp.Niter = Niter;
        cp.Nredo = Nredo;
        cp.Verbose = Verbose ? 1 : 0;
        cp.Spherical = Spherical ? 1 : 0;
        cp.IntCentroids = IntCentroids ? 1 : 0;
        cp.UpdateIndex = UpdateIndex ? 1 : 0;
        cp.FrozenCentroids = FrozenCentroids ? 1 : 0;
        cp.MinPointsPerCentroid = MinPointsPerCentroid;
        cp.MaxPointsPerCentroid = MaxPointsPerCentroid;
        cp.Seed = Seed;
        cp.DecodeBlockSize = (UIntPtr)DecodeBlockSize;
        return cp;
    }
}