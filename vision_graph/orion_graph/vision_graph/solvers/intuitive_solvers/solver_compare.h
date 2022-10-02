#ifndef SOLVER_COMPARE_H_
#define SOLVER_COMPARE_H_

#include "../../include/graph_solver.h"
#include "../../common/utils.h"
#include "../../include/graph_solver_factory.h"
#include "implement/scalar_compare/scalar_compare.h"

namespace vision_graph{

class SolverCompare : public Solver, vision_graph::SolverCreator<SolverCompare>
{   
public:
    SolverCompare(nlohmann::json const& param) noexcept;
    virtual ~SolverCompare() noexcept;

public:
    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;
    
    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    SCALAR_COMPARE_TYPE          compare_type_;
    Tensor*                      const_val_;
    bool                         is_constant_;
};//class SolverLogic

}// namespace vision_graph

#endif