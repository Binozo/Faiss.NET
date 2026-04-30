using Faiss.Cpu.Indexes.Flat;
using Faiss.Cpu.Serializer;
using Xunit;

namespace Faiss.Tests;

/// <summary>
/// Tests index serialization and deserialization via streams and files.
/// </summary>
public class SerializerTests
{
    [Fact]
    public void StreamSerializer_WriteAndRead_RestoresIndexPerfectly()
    {
        int dimensions = 4;
        using var originalIndex = new IndexFlatL2(dimensions);

        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        originalIndex.Add(2, vectors);
        Assert.Equal(2, originalIndex.TotalCount);

        using var memoryStream = new MemoryStream();
        IndexSerializer.Write(originalIndex, memoryStream);

        memoryStream.Position = 0;

        using var restoredIndex = IndexDeserializer.Read<IndexFlatL2>(memoryStream);

        Assert.Equal(dimensions, restoredIndex.Dimensions);
        Assert.Equal(2, restoredIndex.TotalCount);

        float[] query = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        restoredIndex.Search(1, query, 1, distances, labels);

        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }

    [Fact]
    public void FileSerializer_WriteAndRead_RestoresIndexPerfectly()
    {
        int dimensions = 2;
        using var originalIndex = new IndexFlatL2(dimensions);
        originalIndex.Add(1, new float[] { 9.0f, 10.0f });

        string tempFilePath = Path.GetTempFileName();

        try
        {
            IndexSerializer.Write(originalIndex, tempFilePath);
            using var restoredIndex = IndexDeserializer.Read<IndexFlatL2>(tempFilePath);

            Assert.Equal(dimensions, restoredIndex.Dimensions);
            Assert.Equal(1, restoredIndex.TotalCount);
        }
        finally
        {
            if (File.Exists(tempFilePath))
            {
                File.Delete(tempFilePath);
            }
        }
    }
}
