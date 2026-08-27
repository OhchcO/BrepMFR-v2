#include "nx_instance_browser.hpp"

#include "nx_part_adapter.hpp"

#include <NXOpen/Body.hxx>
#include <NXOpen/BodyCollection.hxx>
#include <NXOpen/Face.hxx>
#include <NXOpen/Part.hxx>
#include <NXOpen/PartCollection.hxx>
#include <NXOpen/Session.hxx>

#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace brepmfr
{
namespace nx2212
{
namespace
{
std::unordered_map<FaceKey, NXOpen::Face*> CurrentBodyFacesByTag(
    const PartSnapshot& expected)
{
    ValidateNxCurrentPart(expected);
    NXOpen::Session* session = NXOpen::Session::GetSession();
    if (session == nullptr || session->Parts() == nullptr)
    {
        throw std::runtime_error(
            "NX session or Part collection is unavailable");
    }
    NXOpen::Part* part = session->Parts()->Work();
    if (part == nullptr || part->Bodies() == nullptr ||
        static_cast<FaceKey>(part->Tag()) != expected.part_tag)
    {
        throw std::runtime_error(
            "current Work Part no longer matches recognition");
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
        throw std::runtime_error(
            "current body no longer matches recognition");
    }

    const auto faces = bodies.front()->GetFaces();
    if (faces.size() != expected.face_count)
    {
        throw std::runtime_error(
            "current body face count no longer matches recognition");
    }
    std::unordered_map<FaceKey, NXOpen::Face*> faces_by_tag;
    faces_by_tag.reserve(faces.size());
    for (NXOpen::Face* face : faces)
    {
        if (face == nullptr ||
            !faces_by_tag.emplace(
                static_cast<FaceKey>(face->Tag()), face).second)
        {
            throw std::runtime_error(
                "current body contains a null or duplicate NX face");
        }
    }
    return faces_by_tag;
}

class NxTemporaryFaceHighlighter final : public TemporaryHighlightSink
{
public:
    explicit NxTemporaryFaceHighlighter(PartSnapshot expected)
        : expected_(std::move(expected))
    {
    }

    ~NxTemporaryFaceHighlighter() override
    {
        Clear();
    }

    void Replace(const std::vector<FaceKey>& face_tags) override
    {
        Clear();
        if (face_tags.empty())
        {
            throw std::runtime_error(
                "temporary highlight requires at least one NX face");
        }
        const auto faces_by_tag = CurrentBodyFacesByTag(expected_);
        std::set<FaceKey> unique_tags;
        std::vector<NXOpen::Face*> targets;
        targets.reserve(face_tags.size());
        for (const FaceKey tag : face_tags)
        {
            if (!unique_tags.insert(tag).second)
            {
                throw std::runtime_error(
                    "temporary highlight contains duplicate NX face tags");
            }
            const auto found = faces_by_tag.find(tag);
            if (found == faces_by_tag.end())
            {
                throw std::runtime_error(
                    "temporary highlight references a face outside the current body");
            }
            targets.push_back(found->second);
        }

        try
        {
            for (NXOpen::Face* face : targets)
            {
                face->Highlight();
                highlighted_.push_back(face);
            }
        }
        catch (...)
        {
            Clear();
            throw;
        }
    }

    void Clear() noexcept override
    {
        for (NXOpen::Face* face : highlighted_)
        {
            try
            {
                if (face != nullptr)
                {
                    face->Unhighlight();
                }
            }
            catch (...)
            {
            }
        }
        highlighted_.clear();
    }

private:
    PartSnapshot expected_;
    std::vector<NXOpen::Face*> highlighted_;
};
}
}
}
