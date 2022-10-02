#ifndef SOLVER_SELECT_BOXES_H_
#define SOLVER_SELECT_BOXES_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverSelectBoxes : public Solver, vision_graph::SolverCreator<SolverSelectBoxes>
{   
public:
    SolverSelectBoxes(nlohmann::json const& param) noexcept;
    virtual ~SolverSelectBoxes() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    std::vector<std::string>     filters_;
    float                        score_threshold_;
    float                        area_threshold_;
};//class SolverSelectBoxesImage

}//namespace vision_graph

#endif