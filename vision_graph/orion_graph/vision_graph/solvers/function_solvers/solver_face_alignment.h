#ifndef  SOLVER_FACE_ALIGMENT_H_
#define  SOLVER_FACE_ALIGMENT_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverFaceAlignment : public Solver, vision_graph::SolverCreator<SolverFaceAlignment>
{   
public:
    SolverFaceAlignment(nlohmann::json const& param) noexcept;
    virtual ~SolverFaceAlignment() noexcept;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;
};//class SolverKeypoints

};//namespace vision_graph

#endif