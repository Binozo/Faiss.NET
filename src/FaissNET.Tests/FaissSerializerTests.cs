namespace Faiss.Tests;

using System.IO;
using Xunit;

using Cpu.Indexes;
using Cpu.Serializer;

/// <summary>
/// Proving our I/O architecture is solid and doesn't memory leak.
/// Tests both physical files and pure in-memory cloud streams.
/// </summary>
public class FaissSerializerTests
{
    [Fact]
    public void StreamSerializer_WriteAndRead_RestoresIndexPerfectly()
    {
        // Arrange: Spin up a fresh index and train it with some data
        int dimensions = 4;
        using var originalIndex = new FaissIndexFlatL2(dimensions);
        
        float[] vectors = { 1.0f, 2.0f, 3.0f, 4.0f,   5.0f, 6.0f, 7.0f, 8.0f };
        originalIndex.Add(2, vectors);

        Assert.Equal(2, originalIndex.TotalCount);

        // Act: Serialize directly to a pure .NET MemoryStream (simulating cloud I/O)
        using var memoryStream = new MemoryStream();
        FaissSerializer.Write(originalIndex, memoryStream);

        // Reset the stream position to the beginning so we can actually read it back!
        memoryStream.Position = 0;

        // Deserialize the index directly from the stream using our goated C++ callbacks
        using var restoredIndex = FaissSerializer.Read<FaissIndexFlatL2>(memoryStream);

        // Assert: The restored index should be a perfect clone
        Assert.Equal(dimensions, restoredIndex.Dimensions);
        Assert.Equal(2, restoredIndex.TotalCount);

        // Let's do a quick search to prove the math didn't get cooked during the transfer
        float[] query = { 1.0f, 2.0f, 3.0f, 4.0f };
        float[] distances = new float[1];
        long[] labels = new long[1];

        restoredIndex.Search(1, query, 1, distances, labels);

        // The closest match should still be ID 0 with a distance of 0
        Assert.Equal(0, labels[0]);
        Assert.Equal(0.0f, distances[0]);
    }
    
    [Fact]
    public void FileSerializer_WriteAndRead_RestoresIndexPerfectly()
    {
        // Arrange
        int dimensions = 2;
        using var originalIndex = new FaissIndexFlatL2(dimensions);
        originalIndex.Add(1, new float[] { 9.0f, 10.0f });
        
        // Create a temporary file path so we don't clutter the dev machine
        string tempFilePath = Path.GetTempFileName();

        try
        {
            // Act: Save directly to the SSD
            FaissSerializer.Write(originalIndex, tempFilePath);

            // Load it back from the SSD
            using var restoredIndex = FaissSerializer.Read<FaissIndexFlatL2>(tempFilePath);

            // Assert
            Assert.Equal(dimensions, restoredIndex.Dimensions);
            Assert.Equal(1, restoredIndex.TotalCount);
        }
        finally
        {
            // Clean up the temp file so the hard drive doesn't get completely cooked over time
            if (File.Exists(tempFilePath))
            {
                File.Delete(tempFilePath);
            }
        }
    }
}