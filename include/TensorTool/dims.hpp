#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "seq.hpp"

namespace TensorTool
{
    template <Eigen::DenseIndex Dim, typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>>>
    inline auto squeeze(const Derived &data)
    {
        using Type = Eigen::TensorBase<Derived>;
        static_assert(Dim < Type::NumDimensions && Dim >= -Type::NumDimensions);
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Type::NumDimensions + Dim);
        if (data.dimension(realDim) != 1)
            throw std::invalid_argument("squeeze dim must equal to 1");
        return data.chip(0, realDim);
    }

    template <Eigen::DenseIndex Dim, typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived>, Derived>>>
    inline auto unsqueeze(const Derived &data)
    {
        using Type = Eigen::TensorBase<Derived>;
        static_assert(Dim <= Type::NumDimensions && Dim >= -Type::NumDimensions - 1);
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Type::NumDimensions + Dim + 1);
        auto size = splice<realDim, 0, 1>(data.dimensions(), {1});
        return data.reshape(splice<realDim, 0, 1>(data.dimensions(), {1}));
    }

    namespace internal
    {
        template <Eigen::DenseIndex Dim, typename T, int N, typename Derived, typename... Other>
        inline void stack_impl(int base, Eigen::Tensor<T, N> &res, const Derived &arg0, const Other &...args)
        {
            res.slice(splice<Dim, 1, 1>(Eigen::DSizes<int, N>{}, {base}), splice<Dim, 1, 1>(res.dimensions(), {1})) = unsqueeze<Dim>(arg0);
            if constexpr (sizeof...(args) > 0)
                stack_impl<Dim>(base + 1, res, args...);
        }
    }

    template <Eigen::DenseIndex Dim, typename Derived, typename... Other>
    inline auto stack(const Derived &arg0, const Other &...args)
    {
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Derived::NumDimensions + Dim + 1);
        using T = decltype(unsqueeze<Dim>(arg0));
        Eigen::Tensor<typename T::Scalar, T::NumDimensions> res(splice<realDim, 0, 1>(arg0.dimensions(), {sizeof...(args) + 1}));
        internal::stack_impl<realDim>(0, res, arg0, args...);
        return res;
    }
    // TODO: stack, broadcast-op, etc.
}
