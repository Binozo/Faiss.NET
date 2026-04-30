using System.IO;
using Faiss.Cpu.Extensions;
using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Serializer;

namespace Faiss.Gpu.Tests;

using Xunit;

using Cpu.Indexes;
using Gpu.Resources;

/// <summary>
/// Tests serialization and deserialization round-trips for GPU indexes.
/// </summary>
[Collection("GpuSequential")]
public class GpuSerializationTests
{
    [Fact]
    public void TransferToGpu_RoundTripThroughFile_PreservesVectors()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        string tempFile = Path.GetTempFileName();

        try
        {
            using (var originalCpu = new IndexFlatL2(dimensions))
            {
                float[] vectors =
                {
                    1.0f, 2.0f, 3.0f, 4.0f,
                    5.0f, 6.0f, 7.0f, 8.0f
                };
                originalCpu.Add(2, vectors);

                IndexSerializer.Write(originalCpu, tempFile);
            }

            using (var deserializedCpu = IndexDeserializer.Read<IndexFlatL2>(tempFile))
            using (var resources = new GpuResourcesProvider())
            using (var gpuIndex = GpuIndexProvider.TransferToGpu(resources, deserializedCpu))
            {
                Assert.Equal(2, gpuIndex.TotalCount);

                float[] distances = new float[1];
                long[] labels = new long[1];
                gpuIndex.Search(1, [5.0f, 6.0f, 7.0f, 8.0f], 1, distances, labels);

                Assert.Equal(1, labels[0]);
            }
        }
        finally
        {
            if (File.Exists(tempFile))
            {
                File.Delete(tempFile);
            }
        }
    }

    [Fact]
    public void TransferToGpu_RoundTripThroughMemoryStream_PreservesVectors()
    {
        GpuFact.SkipIfNoGpu();

        int dimensions = 4;
        using var originalCpu = new IndexFlatL2(dimensions);

        float[] vectors =
        {
            1.0f, 2.0f, 3.0f, 4.0f,
            5.0f, 6.0f, 7.0f, 8.0f
        };
        originalCpu.Add(2, vectors);

        using var stream = new MemoryStream();
        IndexSerializer.Write(originalCpu, stream);
        stream.Position = 0;

        using var deserializedCpu = IndexDeserializer.Read<IndexFlatL2>(stream);
        using var resources = new GpuResourcesProvider();
        using var gpuIndex = GpuIndexProvider.TransferToGpu(resources, deserializedCpu);

        Assert.Equal(2, gpuIndex.TotalCount);

        float[] distances = new float[1];
        long[] labels = new long[1];
        gpuIndex.Search(1, [1.0f, 2.0f, 3.0f, 4.0f], 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
}
