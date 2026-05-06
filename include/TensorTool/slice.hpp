#pragma once
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include "dims.hpp"

namespace TensorTool
{
    struct More
    {
    };

    struct Range
    {
        std::optional<int64_t> begin;
        std::optional<int64_t> end;
    };

    namespace internal
    {
        template <typename Derived, typename Dims, typename Res1, typename Res2, typename I0, typename... Is>
        inline bool create_slice_index(const Derived &data, size_t base, Res1 &start_list, Res1 &len_list, Res2 &reverse_list, const Dims &dims, const I0 &arg0, const Is &...args)
        {
            using T = typename Eigen::internal::traits<Derived>::Index;
            T start = 0;
            T len = dims[base];
            bool reverse = false;
            if constexpr (!std::is_base_of_v<I0, More>)
            {
                static_assert(std::is_base_of_v<I0, Range>);
                T begin = arg0.begin.value_or(0);
                T end = arg0.end.value_or(dims[base]);
                begin = (begin >= 0 ? begin : dims[base] + begin);
                end = (end >= 0 ? end : dims[base] + end);
                if (begin < 0 || end < 0 || begin > len || end > len)
                    throw std::invalid_argument("illegal slice range");
                if (end < begin)
                {
                    std::swap(begin, end);
                    begin += 1;
                    end += 1;
                    reverse_list[base] = true;
                    reverse = true;
                }
                start = begin;
                len = end - begin;
            }
            start_list[base] = start;
            len_list[base] = len;
            if constexpr (sizeof...(Is) != 0)
            {
                size_t next_base = base + 1;
                if constexpr (std::is_base_of_v<I0, More>)
                {
                    next_base = dims.size() - sizeof...(Is);
                }
                reverse = create_slice_index(data, next_base, start_list, len_list, reverse_list, dims, args...) || reverse;
            }
            return reverse;
        }
    }

    template <typename Derived, typename I0, typename... Is>
    inline auto slice(const Derived &data, const I0 &arg0, const Is &...args)
    {
        static_assert(sizeof...(args) + 1 <= Eigen::internal::traits<Derived>::NumDimensions);
        if constexpr (sizeof...(Is) > 0)
            static_assert(std::is_base_of_v<I0, More> + (std::is_base_of_v<Is, More> + ...) <= 1);
        using array = std::array<typename Eigen::internal::traits<Derived>::Index, Eigen::internal::traits<Derived>::NumDimensions>;
        auto dims = dimensions(data);
        array start_list{};
        array len_list = dims;
        std::array<bool, Eigen::internal::traits<Derived>::NumDimensions> reverse_list{};
        internal::create_slice_index(data, 0, start_list, len_list, reverse_list, dims, arg0, args...);
        return data.slice(start_list, len_list).reverse(reverse_list);
    }

    template <typename Derived, typename I0, typename... Is>
    inline auto slice_ref(Derived &data, const I0 &arg0, const Is &...args)
    {
        static_assert(sizeof...(args) + 1 <= Eigen::internal::traits<Derived>::NumDimensions);
        if constexpr (sizeof...(Is) > 0)
            static_assert(std::is_base_of_v<I0, More> + (std::is_base_of_v<Is, More> + ...) <= 1);
        using array = std::array<typename Eigen::internal::traits<Derived>::Index, Eigen::internal::traits<Derived>::NumDimensions>;
        auto dims = dimensions(data);
        array start_list{};
        array len_list = dims;
        std::array<bool, Eigen::internal::traits<Derived>::NumDimensions> reverse_list{};
        auto reverse = internal::create_slice_index(data, 0, start_list, len_list, reverse_list, dims, arg0, args...);
        if (reverse)
            throw std::invalid_argument("slice ref does not support reverse-index");
        return data.slice(start_list, len_list);
    }
}