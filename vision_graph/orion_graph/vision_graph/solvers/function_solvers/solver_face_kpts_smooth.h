#ifndef SOLVER_FACE_KPTS_SMOOTH_H_
#define SOLVER_FACE_KPTS_SMOOTH_H_

#include "../../include/graph_solver.h"
#include "../../include/graph_solver_factory.h"
#include "implement/face_kpts_smooth.h"

namespace vision_graph{

class SolverFaceKptsSmooth : public Solver, vision_graph::SolverCreator<SolverFaceKptsSmooth>
{ 
public:
    SolverFaceKptsSmooth(nlohmann::json const& param) noexcept;
    virtual ~SolverFaceKptsSmooth() noexcept;

    static const int                 FACE_SMOOTH_MAX = 100;

    std::string const&               GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                  CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                     CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                      Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept ;

private:
    static vision::Box               get_bounding_box(std::vector<cv::Point2f> const& points) noexcept;
    int                              solve_target(TensorKeypointsVector const*  tensor_kpts_in,
                                                  TensorBoxVector const*        tensor_box_in,
                                                  TensorKeypointsVector*        tensor_kpts_out,
                                                  TensorBoxVector*              tensor_box_out) noexcept ;

protected:
    std::map<int, FaceKptsSmooth*>   face_smooth_map_;
    int                              face_list_[FACE_SMOOTH_MAX];
    int                              face_head_;
    bool                             multi_target_;
};//class SolverFaceKptsSmooth

}//namespace vision_graph

#endif