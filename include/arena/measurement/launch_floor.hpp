#pragma once

namespace arena {

// Median op_ms of an empty kernel, measured the way op_ms is: event, one
// launch, event, synchronize. No kernel can report less, so at small sizes
// it, not bandwidth, is the roof a speed-of-light figure should use.
// Returns 0 if it could not be measured.
float measure_launch_floor_ms();

// An empty launch plus the least real work: one value read from DRAM (L2 was
// evicted just before) and one written. A kernel that reads its input from
// memory cannot finish faster, so this, not the empty launch, is the honest
// roof at tiny sizes. Returns 0 if it could not be measured.
float measure_roundtrip_floor_ms();

}
