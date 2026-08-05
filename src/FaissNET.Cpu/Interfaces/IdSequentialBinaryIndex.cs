namespace Faiss.Cpu.Interfaces;

public interface IIdSequentialBinaryIndex : INativeBinaryIndex
{
    // TODO Add a BinaryIndexIDMap class specifically for binary indexes

    /// <summary>
    /// Adds binary vectors to the index.
    /// </summary>
    /// <param name="count">Number of vectors being added.</param>
    /// <param name="vectors">Packed binary vectors (size: count * Dimensions / 8).</param>
    public void Add(long count, ReadOnlySpan<byte> vectors);
}