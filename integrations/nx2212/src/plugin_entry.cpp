#include "plugin_exports.hpp"
#include "nx_plugin_host.hpp"
#include "plugin_runtime.hpp"

#include <uf_ugopenint.h>

namespace
{
void SetReturnCode(int* retcod, int value) noexcept
{
    if (retcod != nullptr)
    {
        *retcod = value;
    }
}
}

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    SetReturnCode(retcod, brepmfr::nx2212::kFailure);

    try
    {
        brepmfr::nx2212::NxPluginHost host;
        SetReturnCode(
            retcod,
            brepmfr::nx2212::RunInferenceWorkflow(host));
    }
    catch (...)
    {
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
