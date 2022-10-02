#ifndef  SOLVER_FEATURE_H_
#define  SOLVER_FEATURE_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include <reidfeature.h>

namespace vision_graph{

class SolverFeature : public Solver, vision_graph::SolverCreator<SolverFeature>
{   
public:
    SolverFeature(nlohmann::json const& param) noexcept;
    virtual ~SolverFeature() noexcept;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    vision::Feature*               feature_;
    bool                           is_norm_;
};//class SolverFeature

};//namespace vision_graph

#endif