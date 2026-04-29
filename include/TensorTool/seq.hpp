#pragma once
#include <array>
#include <unsupported/Eigen/CXX11/Tensor>

namespace TensorTool
{
    // template <size_t Idx, size_t Del, size_t Ins, typename T, size_t N>
    // std::array<T, N - Del + Ins> splice(const std::array<T, N> &base, const std::array<T, Ins> &ins)
    // {
    //     static_assert(Del <= N, "Del must less or equal N");
    //     std::array<T, N - Del + Ins> res;
    //     if constexpr (Idx > 0)
    //         std::copy(base.begin(), base.begin() + Idx, res.begin());
    //     if constexpr (Ins > 0)
    //         std::copy(ins.begin(), ins.end(), res.begin() + Idx);
    //     if constexpr (Idx < N)
    //         std::copy(base.begin() + Idx + Del, base.end(), res.begin() + Idx + Ins);
    //     return res;
    // }

    template <int Idx, int Del, int Ins, typename T, int N>
    Eigen::DSizes<T, N - Del + Ins> splice(const Eigen::DSizes<T, N> &base, const std::array<T, Ins> &ins)
    {
        static_assert(Del <= N, "Del must less or equal N");
        Eigen::DSizes<T, N - Del + Ins> res;
        if constexpr (Idx > 0)
            std::copy(base.begin(), base.begin() + Idx, res.begin());
        if constexpr (Ins > 0)
            std::copy(ins.begin(), ins.end(), res.begin() + Idx);
        if constexpr (Idx < N)
            std::copy(base.begin() + Idx + Del, base.end(), res.begin() + Idx + Ins);
        return res;
    }
}
