#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <random>
#include <limits>
#include <numeric>
#include <functional>
#include <algorithm>

#include "definitions.hpp"
#include "vector.hpp"

namespace {
    std::default_random_engine rand_gen(12345);
    std::uniform_real_distribution<Real> rand_dist(- 1e20, 1e20);
    std::uniform_int_distribution<int> rand_dim(1, 3);

    auto double_gen = std::bind(rand_dist, rand_gen);
    auto dim_gen    = std::bind(rand_dim, rand_gen);

    const long N_RANDOM_TRIALS = 10000;

    std::vector<Real> random_vec(int dimensions) {
        std::vector<Real> v (dimensions);
        for (int i = 0; i < dimensions; ++i) {
            v[i] = double_gen();
        }
        return v;
    }
}

TEST(Vector, basics) {
    Vector b1(1);
    Vector b3(3);
    b1[0] = 123.45;
    b3[0] = -1;
    b3[2] = 3;

    EXPECT_EQ(123.45, b1[0]);
    EXPECT_EQ(-1,     b3[0]);
    EXPECT_EQ( 0,     b3[1]);
    EXPECT_EQ( 3,     b3[2]);
    EXPECT_EQ( 1, b1.get_dimensions() );
    EXPECT_EQ( 3, b3.get_dimensions() );
}

TEST(Vector, copy_init) {
    Vector a(3);
    a[1] = 123;
    a[2] = -1;
    Vector b = a;
    Vector c = {{0, 123, -1}};
    EXPECT_EQ(a, b);
    EXPECT_EQ(a, c);
}

TEST(Vector, dot_prod) {
    for (int i = 0; i < N_RANDOM_TRIALS; i++) {
        int dims = dim_gen();
        std::vector<Real> a_vec = random_vec(dims);
        std::vector<Real> b_vec = random_vec(dims);
        Vector a (a_vec);
        Vector b (b_vec);

        Real expected = std::inner_product(a_vec.begin(), a_vec.end(), b_vec.begin(), (Real) 0.0);
        EXPECT_DOUBLE_EQ(expected, a * b);
    }
}

TEST(Vector, addSub) {
    for (int i = 0; i < N_RANDOM_TRIALS; i++) {
        int dims = dim_gen();
        std::vector<Real> a_vec = random_vec(dims);
        std::vector<Real> b_vec = random_vec(dims);
        std::vector<Real> c_vec(dims);
        std::vector<Real> d_vec(dims);

        for (int i = 0; i < dims; i++) {
            c_vec[i] = a_vec[i] + b_vec[i];
            d_vec[i] = a_vec[i] - b_vec[i];
        }

        Vector a (a_vec);
        Vector b (b_vec);
        Vector c (c_vec);
        Vector d (d_vec);

        ASSERT_TRUE(c == a + b);
        ASSERT_TRUE(d == a - b);
        ASSERT_TRUE(c == (a += b));
        ASSERT_TRUE(c == a);

        a = { a_vec };

        ASSERT_TRUE(d == (a -= b));
        ASSERT_TRUE(d == a);
    }
}

TEST(Vector, scalarOperations) {
    for (int i = 0; i < N_RANDOM_TRIALS; i++) {
        int dims = dim_gen();
        Real scalar = double_gen();
        Vector a = { random_vec(dims) };
        Vector b = a * scalar * scalar;
        Vector c = scalar + a + scalar;
        Vector d = (- scalar) - a - scalar;
        Vector e = a / scalar;
        for (int i = 0; i < dims; ++i) {
            ASSERT_DOUBLE_EQ(a[i] * square(scalar), b[i]);
            ASSERT_DOUBLE_EQ(scalar + a[i] + scalar, c[i]);
            ASSERT_DOUBLE_EQ(-scalar - a[i] - scalar, d[i]);
            ASSERT_DOUBLE_EQ(a[i] / scalar, e[i]);
        }

        b *= scalar;
        c += scalar;
        d -= scalar;
        e /= scalar;

        for (int i = 0; i < dims; ++i) {
            ASSERT_DOUBLE_EQ(a[i] * square(scalar) * scalar, b[i]);
            ASSERT_DOUBLE_EQ(scalar + a[i] + scalar + scalar, c[i]);
            ASSERT_DOUBLE_EQ(-scalar - a[i] - scalar - scalar, d[i]);
            ASSERT_DOUBLE_EQ(a[i] / scalar / scalar, e[i]);
        }
    }
}

TEST(Vector, display) {
    Vector b = {{ 1, 2, 3 }};
    std::stringstream actual;
    actual << b;
    std::stringstream expected;
    expected << "Vector(1, 2, 3)";
    EXPECT_EQ(expected.str(), actual.str());
}
