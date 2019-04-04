#pragma once

#include "definitions.hpp"
#include "system.hpp"

#include <vector>

namespace Distance
{
void start_tracking(const System&);
void start_tracking(const System&, Real L);
void stop_tracking();
void set_simulation_box_size(Real size);
Real get_simulation_box_size();
void disable_simulation_box();
void invalidate_cache(int i);
Real probe(const System&, int i, int j);

}  // namespace Distance
