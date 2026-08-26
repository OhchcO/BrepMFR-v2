#include "nx_coloring.hpp"

#include "nx_part_adapter.hpp"

#include <NXOpen/Body.hxx>
#include <NXOpen/BodyCollection.hxx>
#include <NXOpen/DisplayManager.hxx>
#include <NXOpen/DisplayModification.hxx>
#include <NXOpen/DisplayableObject.hxx>
#include <NXOpen/Face.hxx>
#include <NXOpen/NXString.hxx>
#include <NXOpen/Part.hxx>
#include <NXOpen/PartCollection.hxx>
#include <NXOpen/Session.hxx>

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
constexpr char kUndoMarkName[] = "BrepMFR feature colors";

struct TargetFace
{
    NXOpen::Face* face;
    int original_color;
};

NXOpen::Session* RequireSession()
{
    NXOpen::Session* session = NXOpen::Session::GetSession();
    if (session == nullptr || session->Parts() == nullptr ||
        session->DisplayManager() == nullptr)
    {
        throw std::runtime_error(
            "NX session, Part collection, or display manager is unavailable");
    }
    return session;
}

std::unordered_map<FaceKey, NXOpen::Face*> CurrentFaceByTag(
    NXOpen::Session* session,
    const PartSnapshot& expected)
{
    NXOpen::Part* part = session->Parts()->Work();
    if (part == nullptr || part->Bodies() == nullptr ||
        static_cast<FaceKey>(part->Tag()) != expected.part_tag)
    {
        throw std::runtime_error("current Work Part no longer matches recognition");
    }

    std::vector<NXOpen::Body*> bodies;
    for (NXOpen::Body* body : *part->Bodies())
    {
        if (body != nullptr)
        {
            bodies.push_back(body);
        }
    }
    if (bodies.size() != 1 ||
        static_cast<FaceKey>(bodies.front()->Tag()) != expected.body_tag)
    {
        throw std::runtime_error("current body no longer matches recognition");
    }

    const auto faces = bodies.front()->GetFaces();
    if (faces.size() != expected.face_count)
    {
        throw std::runtime_error(
            "current body face count no longer matches recognition");
    }
    std::unordered_map<FaceKey, NXOpen::Face*> by_tag;
    by_tag.reserve(faces.size());
    for (NXOpen::Face* face : faces)
    {
        if (face == nullptr ||
            !by_tag.emplace(
                static_cast<FaceKey>(face->Tag()), face).second)
        {
            throw std::runtime_error(
                "current body contains a null or duplicate NX face");
        }
    }
    return by_tag;
}

void ApplyColorGroups(
    NXOpen::Session* session,
    const std::map<int, std::vector<NXOpen::DisplayableObject*>>& groups)
{
    for (const auto& group : groups)
    {
        std::unique_ptr<NXOpen::DisplayModification> modification(
            session->DisplayManager()->NewDisplayModification());
        if (!modification)
        {
            throw std::runtime_error(
                "NX could not create a DisplayModification");
        }
        modification->SetApplyToAllFaces(false);
        modification->SetNewColor(group.first);
        modification->Apply(group.second);
    }
}

std::string RestoreOriginalColors(
    NXOpen::Session* session,
    const std::vector<TargetFace>& targets)
{
    try
    {
        std::map<int, std::vector<NXOpen::DisplayableObject*>> restore_groups;
        for (const auto& target : targets)
        {
            if (target.face->Color() != target.original_color)
            {
                restore_groups[target.original_color].push_back(target.face);
            }
        }
        ApplyColorGroups(session, restore_groups);
        return {};
    }
    catch (const std::exception& error)
    {
        return error.what();
    }
    catch (...)
    {
        return "restoring original NX colors failed with an unknown error";
    }
}

std::string RollBackColoring(
    NXOpen::Session* session,
    NXOpen::Session::UndoMarkId mark,
    const std::vector<TargetFace>& targets)
{
    std::vector<std::string> failures;
    try
    {
        session->UndoToMark(mark, kUndoMarkName);
    }
    catch (const std::exception& error)
    {
        failures.push_back(std::string("UndoToMark: ") + error.what());
    }
    catch (...)
    {
        failures.push_back("UndoToMark: unknown error");
    }

    const std::string restore_failure = RestoreOriginalColors(session, targets);
    if (!restore_failure.empty())
    {
        failures.push_back("explicit color restore: " + restore_failure);
    }
    try
    {
        if (session->DoesUndoMarkExist(mark, kUndoMarkName))
        {
            session->DeleteUndoMark(mark, kUndoMarkName);
        }
    }
    catch (const std::exception& error)
    {
        failures.push_back(std::string("DeleteUndoMark: ") + error.what());
    }
    catch (...)
    {
        failures.push_back("DeleteUndoMark: unknown error");
    }

    std::string diagnostic;
    for (const auto& failure : failures)
    {
        if (!diagnostic.empty())
        {
            diagnostic += "; ";
        }
        diagnostic += failure;
    }
    return diagnostic;
}
}

void ApplyNxFaceColorsAtomically(
    const PartSnapshot& expected,
    const std::vector<FaceColorAssignment>& assignments)
{
    try
    {
        ValidateNxCurrentPart(expected);
        if (assignments.empty())
        {
            return;
        }
        NXOpen::Session* session = RequireSession();
        const auto face_by_tag = CurrentFaceByTag(session, expected);

        std::set<FaceKey> assigned_tags;
        std::map<int, int> color_for_instance;
        std::map<int, std::vector<NXOpen::DisplayableObject*>> color_groups;
        std::vector<TargetFace> targets;
        targets.reserve(assignments.size());
        for (const auto& assignment : assignments)
        {
            if (assignment.instance_id < 0)
            {
                throw std::runtime_error(
                    "NX color assignment has a negative instance ID");
            }
            if (assignment.color_index < 1 || assignment.color_index > 216)
            {
                throw std::runtime_error(
                    "NX color assignment is outside [1, 216]");
            }
            if (!assigned_tags.insert(assignment.nx_face_tag).second)
            {
                throw std::runtime_error(
                    "NX color assignments contain duplicate face tags");
            }
            const auto found = face_by_tag.find(assignment.nx_face_tag);
            if (found == face_by_tag.end())
            {
                throw std::runtime_error(
                    "NX color assignment references a face outside the current body");
            }
            const auto instance = color_for_instance.emplace(
                assignment.instance_id, assignment.color_index);
            if (!instance.second && instance.first->second != assignment.color_index)
            {
                throw std::runtime_error(
                    "one feature instance was assigned multiple NX colors");
            }
            color_groups[assignment.color_index].push_back(found->second);
            targets.push_back({found->second, found->second->Color()});
        }

        const auto mark = session->SetUndoMark(
            NXOpen::Session::MarkVisibilityVisible, kUndoMarkName);
        try
        {
            ApplyColorGroups(session, color_groups);
        }
        catch (const std::exception& error)
        {
            const std::string rollback = RollBackColoring(
                session, mark, targets);
            std::string diagnostic =
                std::string("applying NX face colors failed: ") + error.what();
            if (!rollback.empty())
            {
                diagnostic += "; rollback issues: " + rollback;
            }
            throw std::runtime_error(diagnostic);
        }
        catch (...)
        {
            const std::string rollback = RollBackColoring(
                session, mark, targets);
            std::string diagnostic =
                "applying NX face colors failed with an unknown error";
            if (!rollback.empty())
            {
                diagnostic += "; rollback issues: " + rollback;
            }
            throw std::runtime_error(diagnostic);
        }
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            std::string("ApplyNxFaceColorsAtomically failed: ") + error.what());
    }
    catch (...)
    {
        throw std::runtime_error(
            "ApplyNxFaceColorsAtomically failed with an unknown NX error");
    }
}
}
}
