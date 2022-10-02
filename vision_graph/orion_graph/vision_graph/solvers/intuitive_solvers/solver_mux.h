#ifndef SOLVER_MUX_H_
#define SOLVER_MUX_H_

#include "../../include/graph_solver.h"
#include "../../common/utils.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverMux : public Solver, vision_graph::SolverCreator<SolverMux>
{
public:
    SolverMux(nlohmann::json const& param) noexcept ;
    virtual ~SolverMux() noexcept ;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

};// class SolverMux

}// namespace vision_graph

#endif