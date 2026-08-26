#include "plugin_exports.hpp"
#include "nx_plugin_host.hpp"
#include "plugin_runtime.hpp"

#include <uf.h>
#include <uf_ugopenint.h>

#include <stdexcept>
#include <string>

namespace
{
void SetReturnCode(int* retcod, int value) noexcept
{
    if (retcod != nullptr)
    {
        *retcod = value;
    }
}

class UfSession
{
public:
    UfSession()
    {
        const int code = UF_initialize();
        if (code != 0)
        {
            throw std::runtime_error(
                "UF_initialize failed with code " + std::to_string(code));
        }
        initialized_ = true;
    }

    ~UfSession()
    {
        if (initialized_)
        {
            (void)UF_terminate();
        }
    }

    UfSession(const UfSession&) = delete;
    UfSession& operator=(const UfSession&) = delete;

private:
    bool initialized_ = false;
};
}

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    SetReturnCode(retcod, brepmfr::nx2212::kFailure);

    try
    {
        UfSession uf_session;
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
