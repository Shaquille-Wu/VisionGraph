#ifndef SOLVER_GEOMETRY_OFFSET_H_
#define SOLVER_GEOMETRY_OFFSET_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverGeometryOffset : public Solver, vision_graph::SolverCreator<SolverGeometryOffset>
{   
public:
    SolverGeometryOffset(nlohmann::json const& param) noexcept;
    virtual ~SolverGeometryOffset() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;
};//class SolverGeometryOffset

}//namespace vision_graph

#endif