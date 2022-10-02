#ifndef   SOLVER_TRACKER_KCF_H_
#define   SOLVER_TRACKER_KCF_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/tracker/kcf_tracker/kcf.h"

#include "../../include/graph_tensor.h"
#include <vector>

namespace vision_graph{

class SolverTrackerKCF : public Solver, vision_graph::SolverCreator<SolverTrackerKCF>
{   
public:
    SolverTrackerKCF(nlohmann::json const& param) noexcept;
    virtual ~SolverTrackerKCF() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:

    std::vector<kcf_track::KCF_Tracker *> pool;
    std::map<int, kcf_track::KCF_Tracker *> current;
    bool                         has_target_;
    kcf_track::BBox_c            target_box_;
    float                        zoom_scale_;
    int                          track_side_;
    int                          max_tracker_;
};  //class SolverTrackerKCF

}//namespace vision_graph

#endif