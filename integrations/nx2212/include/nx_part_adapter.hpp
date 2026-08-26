#pragma once

#include "plugin_runtime.hpp"

namespace brepmfr
{
namespace nx2212
{
PartSnapshot CaptureNxCurrentPart();
void ValidateNxCurrentPart(const PartSnapshot& expected);
FaceGraph ReadNxFaceGraph(const PartSnapshot& expected);
}
}
