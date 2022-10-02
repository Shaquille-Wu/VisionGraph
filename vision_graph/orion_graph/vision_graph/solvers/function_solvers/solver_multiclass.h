#ifndef  SOLVER_MULTICLASS_H_
#define  SOLVER_MULTICLASS_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include <multiclass.h>

namespace vision_graph{

class SolverMultiClass : public Solver, vision_graph::SolverCreator<SolverMultiClass>
{   
public:
    SolverMultiClass(nlohmann::json const& param) noexcept;
    virtual ~SolverMultiClass() noexcept;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    vision::Multiclass*            multiclass_;
};//class SolverMultiClass

};//namespace vision_graph

#endif