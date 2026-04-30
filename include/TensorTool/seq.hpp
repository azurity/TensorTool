#pragma once
#include <functional>
#include <unsupported/Eigen/CXX11/Tensor>

namespace TensorTool
{
    template <int Idx, int Del, int Ins, typename T, int N>
    inline Eigen::DSizes<T, N - Del + Ins> splice(const Eigen::DSizes<T, N> &base, const std::array<T, Ins> &ins)
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

    template <int Del, int Ins, typename T, int N>
    inline Eigen::DSizes<T, N - Del + Ins> splice(int idx, const Eigen::DSizes<T, N> &base, const std::array<T, Ins> &ins)
    {
        static_assert(Del <= N, "Del must less or equal N");
        Eigen::DSizes<T, N - Del + Ins> res;
        if (idx > 0)
            std::copy(base.begin(), base.begin() + idx, res.begin());
        if (Ins > 0)
            std::copy(ins.begin(), ins.end(), res.begin() + idx);
        if (idx < N)
            std::copy(base.begin() + idx + Del, base.end(), res.begin() + idx + Ins);
        return res;
    }

    template <typename R, int N, typename Fn, typename... T>
    inline Eigen::DSizes<R, N> zip(const Fn &fn, const Eigen::DSizes<T, N>&... args)
    {
        Eigen::DSizes<R, N> res;
        for (size_t i = 0; i < N; i++)
            res[i] = fn(args[i]...);
        return res;
    }
}
