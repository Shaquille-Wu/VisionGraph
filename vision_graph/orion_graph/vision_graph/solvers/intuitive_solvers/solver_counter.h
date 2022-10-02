#ifndef SOLVER_COUNTER_H_
#define SOLVER_COUNTER_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverCounter : public Solver, vision_graph::SolverCreator<SolverCounter>
{   
public:
    typedef enum tag_counter_type{
        INCREASE = 0,
        DESCREASE,
        COUNTER_TYPE_SUM
    }COUNTER_TYPE;

public:
    SolverCounter(nlohmann::json const& param) noexcept;
    virtual ~SolverCounter() noexcept;

    std::string const&  GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*     CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool        CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int         Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

protected:
    long long int       counter_;
    COUNTER_TYPE        counter_type_;
    int                 cycle_;
};//class SolverCounter

}//namespace vision_graph

#endif