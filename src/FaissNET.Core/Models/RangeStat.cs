namespace Faiss.Models;

public enum RangeStat
{
    RS_minmax = 0,    ///< [min - rs*(max-min), max + rs*(max-min)]
    RS_meanstd = 1,   ///< [mean - std * rs, mean + std * rs]
    RS_quantiles = 2, ///< [Q(rs), Q(1-rs)]
    RS_optim = 3      ///< alternate optimization of reconstruction error
}