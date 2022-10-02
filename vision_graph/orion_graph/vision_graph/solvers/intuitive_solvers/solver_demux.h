#ifndef SOLVER_DEMUX_H_
#define SOLVER_DEMUX_H_

#include "../../include/graph_solver.h"
#include "../../common/utils.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverDeMux : public Solver, vision_graph::SolverCreator<SolverDeMux>
{
public:
    SolverDeMux(nlohmann::json const& param) noexcept ;
    virtual ~SolverDeMux() noexcept ;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

};// class SolverDeMux

}// namespace vision_graph

#endif