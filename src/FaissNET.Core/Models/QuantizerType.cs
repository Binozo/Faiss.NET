namespace Faiss.Models;

/// <summary>
/// Compression levels supported by Faiss.
/// </summary>
public enum QuantizerType
{
    QT_8bit = 0,
    QT_4bit = 1,
    QT_8bit_uniform = 2,
    QT_4bit_uniform = 3,
    QT_fp16 = 4,
    QT_8bit_direct = 5,
    QT_6bit = 6,
    QT_bf16 = 7,
    QT_8bit_direct_signed = 8
}