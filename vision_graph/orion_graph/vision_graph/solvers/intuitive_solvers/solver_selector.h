#ifndef SOLVER_SELECTOR_H_
#define SOLVER_SELECTOR_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/scalar_compare/scalar_compare.h"

namespace vision_graph{

class SolverSelector : public Solver, vision_graph::SolverCreator<SolverSelector>
{ 
public:
    typedef enum tag_select_type{
        VAR_NUM=0,
        FIXED_NUM,
        COMPARE,
        PASS_THROUGH,
        SELECT_TYPE_SUM
    }SELECT_TYPE;

public:
    SolverSelector(nlohmann::json const& param) noexcept;
    virtual ~SolverSelector() noexcept;

    std::string const&     GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*        CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool           CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int            Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

private:
    static void            generate_compare_value(nlohmann::json const& sel_json, Tensor*& compare_value) noexcept ;

protected:
    SELECT_TYPE            select_type_;
    int                    select_idx_;
    SCALAR_COMPARE_TYPE    compare_type_;
    Tensor*                compare_val_;
};//class SolverSelector

}//namespace vision_graph

#endif