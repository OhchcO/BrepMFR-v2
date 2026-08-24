#include "plugin_exports.hpp"

#include <uf_ugopenint.h>

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len)
{
    (void)param;
    (void)param_len;
    if (retcod != nullptr)
    {
        *retcod = 0;
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
