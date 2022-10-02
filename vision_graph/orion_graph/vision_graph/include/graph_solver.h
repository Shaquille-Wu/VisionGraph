/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file graph_solver.h
 * @brief This header file defines basic Solver.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef GRAPH_SOLVER_H_
#define GRAPH_SOLVER_H_

#include <string>
#include <vector>
#include "graph_tensor.h"
#include "json.hpp"

namespace vision_graph
{
/**
 * @brief The Solver object is the base class for computation in graph's nodes.
 *
 * it just a abstract class, so, we did not declare its ctor and dtor explicitly
 * 
 * user should implement its 
 * 
 */
class Solver
{
public:
    /**
     * @brief parameter ctor, it execute the construction with json param
     *
     * @param param parameter in json
     * 
     * @note the derived class should implement it, because the graph will call it to construct Solver's instance
     * 
     */
    Solver(nlohmann::json const& param) noexcept {;};

    /**
     * @brief dtor
     *
     */
    virtual ~Solver() noexcept {;};

    /**
     * @brief get solver's name
     * 
     * @return the solver's name
     * 
     */
    std::string const&                 GetSolverName() const noexcept                          { return solver_name_; };

    /**
     * @brief set solver's name
     * 
     * @return none
     * 
     */
    void                               SetSolverName(std::string const& solver_name) noexcept  { solver_name_ = solver_name; } ;

     /**
     * @brief get solver's class name, its a pure-virtual function, so, its derived class should implement it
     * 
     * @return the solver's class name
     * 
     */   
    virtual std::string const&         GetSolverClassName() const noexcept = 0;


     /**
     * @brief set solver's friends list
     * 
     * @param the friends list
     * 
     * @return none
     * 
     */
    virtual void                       SetFriendSolvers(std::vector<Solver*> const& friends) noexcept   { friends_ = friends ; } ;

     /**
     * @brief get solver's friends list
     * 
     * @return the friend solvers list
     * 
     */
    virtual std::vector<Solver*>       GetFriendSolvers() noexcept                                      { return friends_ ; } ;

    /**
     * @brief allocate tensor data, its a pure-virtual function, so, its derived class should implement it
     *
     * @param tensor_name the idx of out tensor data
     * 
     */
    virtual Tensor*                    CreateOutTensor(int out_tensor_idx) noexcept = 0;

    /**
     * @brief check solver's in&out tensor, its a pure-virtual function, so, its derived class should implement it
     *
     * @param in solver's input tensor
     * 
     * @param out solver's output tensor
     * 
     * @return true if all of tensors are valid, other is false
     * 
     */
    virtual bool                       CheckTensor(std::vector<Tensor*> const& in, std::vector<Tensor*> const& out) const noexcept = 0;

    /**
     * @brief execute the computation, its a pure-virtual function, so, its derived class should implement it
     *
     * @param in the input data
     * 
     * @param out the output data
     * 
     * @return the status_code for solve
     * 
     */
    virtual int                        Solve(std::vector<Tensor*> const& in, std::vector<Tensor*>& out) noexcept = 0;

protected:
    std::string                        solver_name_;

    std::vector<Solver*>               friends_;
};

} //namespace vision_graph

#endif