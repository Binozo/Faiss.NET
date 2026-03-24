namespace Faiss.Gpu.Tests;

using Gpu;
using Xunit;
using Models;
using Cpu.Indexes;

public class GpuIndexTests
{
    [Fact]
    public void TransferToGpu_AndSearch_ReturnsExactMatch()
    {
        int dimensions = 4;
        using var cpuIndex = new FaissIndexFlatL2(dimensions);
        Assert.Equal(MetricType.L2, cpuIndex.Metric);
        
        using var gpuContext = new FaissGpuContext();
        gpuContext.SetTempMemory(50 * 1024 * 1024); // 50MB

        using var gpuIndex = GpuIndexProvider.TransferToGpu(gpuContext, cpuIndex, 0);

        Assert.Equal(dimensions, gpuIndex.Dimensions);
        Assert.Equal(0, gpuIndex.TotalCount);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f };
        
        gpuIndex.Add(1, vectors);
        
        Assert.Equal(1, gpuIndex.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];

        gpuIndex.Search(1, vectors, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);

        using var recoveredCpuIndex = GpuIndexProvider.TransferToCpu(gpuIndex);
        Assert.Equal(dimensions, recoveredCpuIndex.Dimensions);
        Assert.Equal(1, recoveredCpuIndex.TotalCount);
        Assert.Equal(MetricType.L2, recoveredCpuIndex.Metric);
    }
}