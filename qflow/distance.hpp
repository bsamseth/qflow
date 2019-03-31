#pragma once

#include "definitions.hpp"
#include "system.hpp"

#include <vector>

namespace Distance
{
void start_tracking(const System&);
void stop_tracking();
Real probe(const System&, int i, int j);
Real invalidate_cache(int i);

}  // namespace Distance
