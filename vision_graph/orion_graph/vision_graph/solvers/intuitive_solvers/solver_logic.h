#ifndef SOLVER_LOGIC_H_
#define SOLVER_LOGIC_H_

#include "../../include/graph_solver.h"
#include "../../common/utils.h"
#include "../../include/graph_solver_factory.h"

namespace vision_graph{

class SolverLogic : public Solver, vision_graph::SolverCreator<SolverLogic>
{   
public:
    typedef enum tag_logic_op_type
    {
        AND = 0,
        OR,
        XOR,
        NOT,
        LOGIC_TYPE_SUM
    }LOGIC_OP_TYPE;

    using LOGIC_RESULT_TYPE = unsigned int;

public:
    SolverLogic(nlohmann::json const& param) noexcept;
    virtual ~SolverLogic() noexcept;

    std::string const&             GetSolverClassName() const noexcept                { return solver_class_name_ ;};

    virtual Tensor*                CreateOutTensor(int out_tensor_idx) noexcept;

    virtual bool                   CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept;

    virtual int                    Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept;

private:
    bool                           get_numeric_logic_value(Tensor const& numeric) noexcept;
    LOGIC_RESULT_TYPE              get_numeric_bitwise_value(Tensor const& numeric) noexcept;
    bool                           logic_op1(Tensor const& oprand, LOGIC_OP_TYPE op_type) noexcept;
    bool                           logic_op2(Tensor const& left, Tensor const& right, LOGIC_OP_TYPE op_type) noexcept;
    LOGIC_RESULT_TYPE              logic_bitwise_op1(Tensor const& oprand, LOGIC_OP_TYPE op_type) noexcept;
    LOGIC_RESULT_TYPE              logic_bitwise_op2(Tensor const& left, Tensor const& right, LOGIC_OP_TYPE op_type) noexcept;

    template<typename IntType, typename std::enable_if<sizeof(IntType) == 4, void>::type* = nullptr>
    IntType                        float2Int(Tensor const& numeric) noexcept
    {
        int  res =  (FLT2INT32((dynamic_cast<TensorFloat32 const*>(&numeric))->value_));
        return (IntType)(res);
    };
    template<typename IntType, typename std::enable_if<sizeof(IntType) == 8, void>::type* = nullptr>
    IntType                        float2Int(Tensor const& numeric) noexcept
    {
        long long int  res =  (FLT2INT64((dynamic_cast<TensorFloat32 const*>(&numeric))->value_));
        return (IntType)(res);
    };
    template<typename IntType, typename std::enable_if<sizeof(IntType) == 4, void>::type* = nullptr>
    IntType                        double2Int(Tensor const& numeric) noexcept
    {
        int  res =  (DBL2INT32((dynamic_cast<TensorFloat64 const*>(&numeric))->value_));

        return (IntType)(res);
    };
    template<typename IntType, typename std::enable_if<sizeof(IntType) == 8, void>::type* = nullptr>
    IntType                        double2Int(Tensor const& numeric) noexcept
    {
        long long int  res =  (DBL2INT64((dynamic_cast<TensorFloat64 const*>(&numeric))->value_));
        return (IntType)(res);
    };

protected:
    LOGIC_OP_TYPE                  logic_type_;
    bool                           bitwise_;
};//class SolverLogic

}// namespace vision_graph

#endif