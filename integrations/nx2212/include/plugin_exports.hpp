#pragma once

#define BREPMFR_NX_EXPORT extern "C" __declspec(dllexport)

BREPMFR_NX_EXPORT void ufusr(char* param, int* retcod, int param_len);
BREPMFR_NX_EXPORT int ufusr_ask_unload();
