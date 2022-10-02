#ifndef  SOLVER_KEYPOINTS_H_
#define  SOLVER_KEYPOINTS_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include <keypoints_attributes.h>

namespace vision_graph{

class SolverKeypoints : public Solver, vision_graph::SolverCreator<SolverKeypoints>
{   
public:
    SolverKeypoints(nlohmann::json const& param) noexcept;
    virtual ~SolverKeypoints() noexcept;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    vision::KeypointsAttributes*   keypoints_;
    bool                           reorder_kpts_;
    float                          angle_scale_;
};//class SolverKeypoints

};//namespace vision_graph

#endif