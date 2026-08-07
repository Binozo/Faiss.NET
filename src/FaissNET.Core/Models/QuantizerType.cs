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
    QT_8bit_direct_signed = 8,
    QT_0bit = 9,
    QT_1bit_tqmse = 10,
    QT_2bit_tqmse = 11,
    QT_3bit_tqmse = 12,
    QT_4bit_tqmse = 13,
    QT_8bit_tqmse = 14,
    QT_2bit_tq = 15,
    QT_3bit_tq = 16,
    QT_4bit_tq = 17,
    QT_5bit_tq = 18,
    QT_1bit_eden = 19,
    QT_2bit_eden = 20,
    QT_3bit_eden = 21,
    QT_4bit_eden = 22,
    QT_5bit_eden = 23,
    QT_6bit_eden = 24,
    QT_7bit_eden = 25,
    QT_8bit_eden = 26
}