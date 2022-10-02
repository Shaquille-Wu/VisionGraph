#ifndef SOLVER_GET_GOOD_BOXES_H_
#define SOLVER_GET_GOOD_BOXES_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverGetGoodBoxes : public Solver, vision_graph::SolverCreator<SolverGetGoodBoxes>
{   
public:
    SolverGetGoodBoxes(nlohmann::json const& param) noexcept;
    virtual ~SolverGetGoodBoxes() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    float                        value_;
};  //class SolverGetGoodBoxes

}//namespace vision_graph

#endif