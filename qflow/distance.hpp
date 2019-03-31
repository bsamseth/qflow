#pragma once

#include "definitions.hpp"
#include "system.hpp"

#include <vector>

namespace Distance
{
void start_tracking(const System&);
void stop_tracking();
void invalidate_cache(int i);
Real probe(const System&, int i, int j);

}  // namespace Distance
