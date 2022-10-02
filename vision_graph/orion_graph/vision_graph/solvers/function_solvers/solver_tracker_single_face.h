#ifndef   SOLVER_TRACKER_SINGLE_FACE_H_
#define   SOLVER_TRACKER_SINGLE_FACE_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/tracker/kcf_tracker/kcf.h"

#include "../../include/graph_tensor.h"
#include "vision_graph.h"
#include <vector>
#include <face_ecotrack_strategy.h>
#include <feature_factory.h>
#include <base_detector.h>
#include <base_keypoints.h>
#include <base_verifier.h>
#include <feat_factory_creator.h>

namespace vision_graph{

class SolverTrackerSingleFace : public Solver, vision_graph::SolverCreator<SolverTrackerSingleFace>
{   
public:
    SolverTrackerSingleFace(nlohmann::json const& param) noexcept;
    virtual ~SolverTrackerSingleFace() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    FeatureFactory* feat_factory = NULL;
    BaseDetector* fp_detector = NULL;
    BaseVerifier* face_verifier = NULL;
    BaseKeypoints* face_keypoints = NULL;
    FaceEcoTrackStrategy* face_tracker = NULL;
    
};  //class SolverTrackerSingleFace

}//namespace vision_graph

#endif  