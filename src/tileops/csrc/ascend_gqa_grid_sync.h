#pragma once

// MIX 1:2 hardware barrier. Launch requires batch scheduling and a grid
// no larger than the physical Cube count. All event tokens must be drained.
#include "kernel_operator.h"

namespace tileops {
__aicore__ inline void gqa_grid_sync() { AscendC::SyncAll<false>(); }
}  // namespace tileops
