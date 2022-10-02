#ifndef SOLVER_DATA_H_
#define SOLVER_DATA_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverData : public Solver, vision_graph::SolverCreator<SolverData>
{   
public:
    SolverData(nlohmann::json const& param) noexcept ;
    virtual ~SolverData() noexcept;

    std::string const&  GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*     CreateOutTensor(int out_tensor_idx) noexcept ;

    virtual bool        CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int         Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept 
    { 
        return 0; 
    };

protected:
    TENSOR_TYPE         data_type_;
    Tensor*             init_value_;
};//class SolverData

}//namespace vision_graph

#endif