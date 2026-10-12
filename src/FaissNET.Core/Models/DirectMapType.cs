namespace Faiss.Models;

public enum DirectMapType : int
{
    /// <summary>
    /// Default
    /// </summary>
    NoMap = 0,
    /// <summary>
    /// Sequential ids (<c>add</c> only)
    /// </summary>
    Array = 1,
    /// <summary>
    /// Arbitrary ids
    /// </summary>
    Hashtable = 2,
}