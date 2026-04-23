namespace Faiss.Models;

[Flags]
public enum IoFlags
{
    None = 0,
    Mmap = 1,        // FAISS_IO_FLAG_MMAP
    ReadOnly = 2,    // FAISS_IO_FLAG_READ_ONLY
}