#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include "seq.hpp"

namespace TensorTool
{
    namespace internal
    {
        template <typename Derived>
        struct dimension_trait
        {
        };

        template <typename Scalar_, int NumIndices_, int Options_, typename IndexType_>
        struct dimension_trait<Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>>
        {
            using T = Eigen::Tensor<Scalar_, NumIndices_, Options_, IndexType_>;
            inline auto dimensions(const T &value) const
            {
                return value.dimensions();
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return value.dimension(N);
            }
        };

        template <typename NullaryOp, typename XprType>
        struct dimension_trait<Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>>
        {
            using T = Eigen::TensorCwiseNullaryOp<NullaryOp, XprType>;
            using SubType = std::remove_cv_t<std::remove_reference_t<XprType>>;
            inline auto dimensions(const T &value) const
            {
                return dimension_trait<SubType>{}.dimensions(value.nestedExpression());
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return dimension_trait<SubType>{}.dimension(value.nestedExpression(), N);
            }
        };

        template <typename UnaryOp, typename XprType>
        struct dimension_trait<Eigen::TensorCwiseUnaryOp<UnaryOp, XprType>>
        {
            using T = Eigen::TensorCwiseUnaryOp<UnaryOp, XprType>;
            using SubType = std::remove_cv_t<std::remove_reference_t<XprType>>;
            inline auto dimensions(const T &value) const
            {
                return dimension_trait<SubType>{}.dimensions(value.nestedExpression());
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return dimension_trait<SubType>{}.dimension(value.nestedExpression(), N);
            }
        };

        template <typename BinaryOp, typename LhsXprType, typename RhsXprType>
        struct dimension_trait<Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>>
        {
            using T = Eigen::TensorCwiseBinaryOp<BinaryOp, LhsXprType, RhsXprType>;
            using SubType = std::remove_cv_t<std::remove_reference_t<LhsXprType>>;
            inline auto dimensions(const T &value) const
            {
                return dimension_trait<SubType>{}.dimensions(value.lhsExpression());
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return dimension_trait<SubType>{}.dimension(value.nestedExpression(), N);
            }
        };

        template <typename Broadcast, typename XprType>
        struct dimension_trait<Eigen::TensorBroadcastingOp<Broadcast, XprType>>
        {
            using T = Eigen::TensorBroadcastingOp<Broadcast, XprType>;
            using SubType = std::remove_cv_t<std::remove_reference_t<XprType>>;
            inline auto dimensions(const T &value) const
            {
                return zip<int>([](auto a, auto b) { return a * b; },
                    dimension_trait<SubType>{}.dimensions(value.expression()), value.broadcast());
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return dimension_trait<SubType>{}.dimension(value.expression(), N) * value.broadcast()[N];
            }
        };

        template <Eigen::DenseIndex DimId, typename XprType>
        struct dimension_trait<Eigen::TensorChippingOp<DimId, XprType>>
        {
            using T = Eigen::TensorChippingOp<DimId, XprType>;
            using SubType = std::remove_cv_t<std::remove_reference_t<XprType>>;
            inline auto dimensions(const T &value) const
            {
                return splice<1, 0>(value.dim(), dimension_trait<SubType>{}.dimensions(value.expression()), {});
            }

            inline auto dimension(const T &value, size_t N) const
            {
                if (N < value.dim())
                    return dimension_trait<SubType>{}.dimension(value.expression(), N);
                else
                    return dimension_trait<SubType>{}.dimension(value.expression(), N - 1);
            }
        };

        template <typename NewDimensions, typename XprType>
        struct dimension_trait<Eigen::TensorReshapingOp<NewDimensions, XprType>>
        {
            using T = Eigen::TensorReshapingOp<NewDimensions, XprType>;
            inline auto dimensions(const T &value) const
            {
                return value.dimensions();
            }

            inline auto dimension(const T &value, size_t N) const
            {
                return value.dimensions()[N];
            }
        };
    }

    template <typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>>
    inline auto dimensions(const Derived &data)
    {
        return internal::dimension_trait<Derived>{}.dimensions(data);
    }

    template <typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>>
    inline auto dimension(const Derived &data, size_t N)
    {
        return internal::dimension_trait<Derived>{}.dimension(data, N);
    }

    template <Eigen::DenseIndex Dim, typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>>
    inline auto squeeze(const Derived &data)
    {
        using Type = Eigen::TensorBase<Derived>;
        static_assert(Dim < Type::NumDimensions && Dim >= -Type::NumDimensions);
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Type::NumDimensions + Dim);
        if (dimension(data, realDim) != 1)
            throw std::invalid_argument("squeeze dim must equal to 1");
        return data.chip(0, realDim);
    }

    template <Eigen::DenseIndex Dim, typename Derived, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>>
    inline auto unsqueeze(const Derived &data)
    {
        using Type = Eigen::TensorBase<Derived>;
        static_assert(Dim <= Type::NumDimensions && Dim >= -Type::NumDimensions - 1);
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Type::NumDimensions + Dim + 1);
        auto size = splice<realDim, 0, 1>(dimensions(data), {1});
        return data.reshape(splice<realDim, 0, 1>(dimensions(data), {1}));
    }

    namespace internal
    {
        template <Eigen::DenseIndex Dim, typename T, int N, typename Derived, typename... Other, typename C = std::enable_if_t<std::is_base_of_v<Eigen::TensorBase<Derived, Eigen::ReadOnlyAccessors>, Derived>>>
        inline void stack_impl(int base, Eigen::Tensor<T, N> &res, const Derived &arg0, const Other &...args)
        {
            res.slice(splice<Dim, 1, 1>(Eigen::DSizes<int, N>{}, {base}), splice<Dim, 1, 1>(dimensions(res), {1})) = unsqueeze<Dim>(arg0);
            if constexpr (sizeof...(args) > 0)
                stack_impl<Dim>(base + 1, res, args...);
        }
    }

    template <Eigen::DenseIndex Dim, typename Derived, typename... Other>
    inline auto stack(const Derived &arg0, const Other &...args)
    {
        constexpr Eigen::DenseIndex realDim = (Dim >= 0 ? Dim : Derived::NumDimensions + Dim + 1);
        using T = decltype(unsqueeze<Dim>(arg0));
        Eigen::Tensor<typename T::Scalar, T::NumDimensions> res(splice<realDim, 0, 1>(dimensions(arg0), {sizeof...(args) + 1}));
        internal::stack_impl<realDim>(0, res, arg0, args...);
        return res;
    }
}
