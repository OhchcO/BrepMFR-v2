#pragma once

#include "plugin_runtime.hpp"

#include <vector>

namespace brepmfr
{
namespace nx2212
{
void ApplyNxFaceColorsAtomically(
    const PartSnapshot& expected,
    const std::vector<FaceColorAssignment>& assignments);
}
}
