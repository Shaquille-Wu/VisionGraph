#ifndef SOLVER_CUT_IMAGE_H_
#define SOLVER_CUT_IMAGE_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverCutImage : public Solver, vision_graph::SolverCreator<SolverCutImage>
{   
public:
    SolverCutImage(nlohmann::json const& param) noexcept;
    virtual ~SolverCutImage() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    bool                         padding_;
};//class SolverSelectBoxImage

}//namespace vision_graph

#endif