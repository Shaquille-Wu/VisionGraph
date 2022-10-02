#ifndef  SOLVER_DETECTOR_H_
#define  SOLVER_DETECTOR_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include <detector.h>

namespace vision_graph{

class SolverDetect : public Solver, vision_graph::SolverCreator<SolverDetect>
{   
public:
    SolverDetect(nlohmann::json const& param) noexcept;
    virtual ~SolverDetect() noexcept;

    std::string const&  GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*     CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool        CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int         Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    vision::Detector*   detector_;
};//class SolverDetect

};//namespace vision_graph

#endif