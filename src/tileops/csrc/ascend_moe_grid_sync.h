#pragma once
#include "kernel_operator.h"
namespace tileops {
__aicore__ inline void moe_grid_sync() { AscendC::SyncAll<true>(); }
}  // namespace tileops
