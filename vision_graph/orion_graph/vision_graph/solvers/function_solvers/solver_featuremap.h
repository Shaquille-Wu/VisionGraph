#ifndef  SOLVER_FEATURE_MAP_H_
#define  SOLVER_FEATURE_MAP_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include <logging.h>

namespace vision_graph{

class SolverFeatureMap : public Solver, vision_graph::SolverCreator<SolverFeatureMap>
{   
public:
    SolverFeatureMap(nlohmann::json const& param) noexcept;
    virtual ~SolverFeatureMap() noexcept;

    std::string const&            GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*               CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                  CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                   Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    vision::FeaturemapRunner*     feature_map_;
};//class SolverFeatureMap

};//namespace vision_graph

#endif