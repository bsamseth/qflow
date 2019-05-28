#include "inputsorter.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#define GET_SORTED(s)                  \
    System sorted(s.rows(), s.cols()); \
    auto   mapping = sort(s);          \
    for (int i = 0; i < s.rows(); ++i) \
        sorted.row(i) = s.row(mapping[i]);

namespace
{
std::vector<int> sort(const System& s)
{
    std::vector<int> mapping(s.rows());
    std::iota(mapping.begin(), mapping.end(), 0);  // Fill with 0, 1, ..., n-1

    Vector dists = s.rowwise().squaredNorm();
    std::sort(mapping.begin(), mapping.end(), [&](int i, int j) {
        return dists[i] < dists[j];
    });

    return mapping;
}
}  // namespace

Real InputSorter::operator()(const System& s)
{
    GET_SORTED(s);
    return f->operator()(sorted);
}

RowVector InputSorter::gradient(const System& s)
{
    GET_SORTED(s);
    return f->gradient(sorted);
}
Real InputSorter::drift_force(const System& s, int k, int d)
{
    GET_SORTED(s);
    return f->drift_force(sorted, mapping[k], d);
}
Real InputSorter::laplacian(const System& s)
{
    GET_SORTED(s);
    return f->laplacian(sorted);
}
