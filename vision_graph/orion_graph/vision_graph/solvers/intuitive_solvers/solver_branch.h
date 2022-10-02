#ifndef SOLVER_BRANCH_H_
#define SOLVER_BRANCH_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/scalar_compare/scalar_compare.h"

namespace vision_graph{

class SolverBranch : public Solver, vision_graph::SolverCreator<SolverBranch>
{       
public:
    typedef enum forward_type{
        VAR_NUM=0,
        VAR_MASK,
        FIXED_NUM,
        FIXED_MASK,
        COMPARE,
    }FORWARD_TYPE;

public:
    SolverBranch(nlohmann::json const& param) noexcept;
    virtual ~SolverBranch() noexcept;

public:
    std::string const&           GetSolverClassName() const noexcept { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;
    
    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;
    
protected:
    FORWARD_TYPE                 forward_type_;
    SCALAR_COMPARE_TYPE          compare_type_;
    Tensor*                      compare_val_;
};//class SolverBranch

}//namespace vision_graph

#endif