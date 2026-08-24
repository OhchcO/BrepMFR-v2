#include "plugin_exports.hpp"
#include "plugin_runtime.hpp"

#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/UI.hxx>
#include <uf_ugopenint.h>

#include <stdexcept>

namespace
{
class NxLoadNotifier final : public brepmfr::nx2212::LoadNotifier
{
public:
    void ShowInformation(const char* title, const char* message) override
    {
        NXOpen::UI* ui = NXOpen::UI::GetUI();
        if (ui == nullptr || ui->NXMessageBox() == nullptr)
        {
            throw std::runtime_error("NX UI is unavailable.");
        }
        ui->NXMessageBox()->Show(
            title,
            NXOpen::NXMessageBox::DialogTypeInformation,
            message);
    }

    void ShowError(const char* title, const char* message) noexcept override
    {
        try
        {
            NXOpen::UI* ui = NXOpen::UI::GetUI();
            if (ui != nullptr && ui->NXMessageBox() != nullptr)
            {
                ui->NXMessageBox()->Show(
                    title,
                    NXOpen::NXMessageBox::DialogTypeError,
                    message);
            }
        }
        catch (...)
        {
        }
    }
};

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
        NxLoadNotifier notifier;
        SetReturnCode(retcod, brepmfr::nx2212::RunLoadConfirmation(notifier));
    }
    catch (...)
    {
    }
}

BREPMFR_NX_EXPORT int ufusr_ask_unload()
{
    return UF_UNLOAD_IMMEDIATELY;
}
