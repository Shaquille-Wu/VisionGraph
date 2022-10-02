#ifndef SOLVER_FACE_HOG_VERIFY_H_
#define SOLVER_FACE_HOG_VERIFY_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/face_hog_verify.h"

namespace vision_graph{

class SolverFaceHOGVerify : public Solver, vision_graph::SolverCreator<SolverFaceHOGVerify>
{   
public:
    typedef enum tag_target_source_type{
        FROM_DEFAULT = 0,
        FROM_FILE,
        FROM_VARIABLE,
        SOURCE_TYPE_SUM
    }TARGET_SOURCE_TYPE;

public:
    SolverFaceHOGVerify(nlohmann::json const& param) noexcept;
    virtual ~SolverFaceHOGVerify() noexcept;

    std::string const&           GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*              CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                 CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                  Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

protected:
    TARGET_SOURCE_TYPE           target_src_type_;
    FaceHOGVerify*               hog_verify_;
};//class SolverFaceHOGVerify

}//namespace vision_graph

#endif