#pragma once
#include <cmath>
#include <tuple>
#include <unsupported/Eigen/CXX11/Tensor>
#include "dims.hpp"

namespace TensorTool
{
    namespace internal
    {
        inline std::tuple<int, int> broadcast_factor(int lhs, int rhs)
        {
            if (lhs > rhs)
            {
                if (lhs % rhs != 0)
                    throw std::invalid_argument("cannot broadcast");
                return {1, lhs / rhs};
            }
            else
            {
                if (rhs % lhs != 0)
                    throw std::invalid_argument("cannot broadcast");
                return {rhs / lhs, 1};
            }
        }

        template <typename T1, typename T2>
        inline auto cwise_broadcast(const T1 &lhs, const T2 &rhs)
        {
            using LHS = Eigen::TensorBase<T1>;
            using RHS = Eigen::TensorBase<T2>;
            static_assert(LHS::NumDimensions == LHS::NumDimensions);
            constexpr int dim = LHS::NumDimensions;
            auto l_size = dimensions(lhs);
            auto r_size = dimensions(rhs);
            for (int i = 0; i < dim; i++)
            {
                std::tie(l_size[i], r_size[i]) = broadcast_factor(l_size[i], r_size[i]);
            }
            return std::make_tuple(l_size, r_size);
        }
    }

#define CWISE_MEMBER_OP(name, op)                                                                                                                                                                                        \
    template <typename T1, typename T2, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<T1, Eigen::ReadOnlyAccessors>, T1> && std::is_base_of_v<Eigen::TensorBase<T2, Eigen::ReadOnlyAccessors>, T2>>> \
    inline auto name(const T1 &lhs, const T2 &rhs)                                                                                                                                                                       \
    {                                                                                                                                                                                                                    \
        auto [l_size, r_size] = internal::cwise_broadcast(lhs, rhs);                                                                                                                                                     \
        return lhs.broadcast(l_size).op(rhs.broadcast(r_size));                                                                                                                                                          \
    }

    CWISE_MEMBER_OP(cwise_add, operator+)
    CWISE_MEMBER_OP(cwise_sub, operator-)
    CWISE_MEMBER_OP(cwise_mul, operator*)
    CWISE_MEMBER_OP(cwise_div, operator/)

#undef CWISE_MEMBER_OP

#define CWISE_UNARY_OP(name, fn)                                                                                                                 \
    template <typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>> \
    inline auto name(const Derived &arg)                                                                                                         \
    {                                                                                                                                            \
        using T = Eigen::TensorBase<Derived>;                                                                                                    \
        return arg.unaryExpr([](const typename T::Scalar &x) { return fn(x); });                                                                 \
    }

    CWISE_UNARY_OP(sin, std::sin)
    CWISE_UNARY_OP(cos, std::cos)
    CWISE_UNARY_OP(tan, std::tan)
    CWISE_UNARY_OP(asin, std::asin)
    CWISE_UNARY_OP(acos, std::acos)
    CWISE_UNARY_OP(atan, std::atan)

#undef CWISE_UNARY_OP
}
