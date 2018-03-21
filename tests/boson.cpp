#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <functional>
#include <algorithm>

#include "definitions.hpp"
#include "boson.hpp"

std::default_random_engine rand_gen(12345);
std::uniform_real_distribution<Real> rand_dist(- 1e20, 1e20);
std::uniform_int_distribution<int> rand_dim(1, 3);

auto double_gen = std::bind(rand_dist, rand_gen);
auto dim_gen    = std::bind(rand_dim, rand_gen);

TEST(Boson, basics) {
    Boson b1(1);
    Boson b3(3);
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

TEST(Boson, copy_init) {
    Boson a(3);
    a[1] = 123;
    a[2] = -1;
    Boson b = a;
    Boson c = {{0, 123, -1}};

    EXPECT_EQ(a, b);
    EXPECT_EQ(a, c);
}

TEST(Boson, dot_prod) {
    for (int i = 0; i < 10000; i++) {
        int dims = dim_gen();
        std::vector<Real> a_vec(dims);
        std::vector<Real> b_vec(dims);

        for (int i = 0; i < dims; i++) {
            a_vec[i] = rand_dist(rand_gen);
            b_vec[i] = rand_dist(rand_gen);
        }

        Boson a (a_vec);
        Boson b (b_vec);

        Real expected = std::inner_product(a_vec.begin(), a_vec.end(), b_vec.begin(), (Real) 0.0);
        EXPECT_NEAR(expected, a * b, 1e-15);
    }
}

TEST(Boson, add_sub) {
    for (int i = 0; i < 10000; i++) {
        int dims = dim_gen();
        std::vector<Real> a_vec(dims);
        std::vector<Real> b_vec(dims);
        std::vector<Real> c_vec(dims);
        std::vector<Real> d_vec(dims);

        for (int i = 0; i < dims; i++) {
            a_vec[i] = rand_dist(rand_gen);
            b_vec[i] = rand_dist(rand_gen);
            c_vec[i] = a_vec[i] + b_vec[i];
            d_vec[i] = a_vec[i] - b_vec[i];
        }

        Boson a (a_vec);
        Boson b (b_vec);
        Boson c (c_vec);
        Boson d (d_vec);

        EXPECT_TRUE(c == a + b);
        EXPECT_TRUE(d == a - b);
    }
}




