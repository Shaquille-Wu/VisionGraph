/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file graph_node.h
 * @brief This header file defines graph node.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef GRAPH_NODE_H_
#define GRAPH_NODE_H_

#include "include/graph_solver.h"

namespace vision_graph
{

/**
 * @brief The Message object is the base class for message between in graph's nodes.
 * 
 * it is just a signal, it just transfer communication's source(msg_id, or say node's id) and status(msg_status)
 * 
 */
class Message
{
public:
    /**
     * @brief Constructor
     *
     */
    Message() noexcept :msg_id_(0), msg_status_(0), msg_tensor_(nullptr) { ; };

    /**
     * @brief Construct with parameter
     *
     * @param msg_id message id, you can specify it with node's id
     * @param msg_status you can specify it with "Solve"'s return value
     * @param msg_tensor msg tensor pointer
     */
    Message(int msg_id, int msg_status, Tensor* msg_tensor) noexcept :msg_id_(msg_id), msg_status_(msg_status), msg_tensor_(msg_tensor) {;};

    /**
     * @brief dtor
     *
     */
    virtual ~Message() noexcept { ; };

    /**
     * @brief assignment constructor
     * 
     * @param other The other Message object.
     * 
     * @exceptsafe No throw.
     */
    Message& operator=(Message const& other) noexcept
    {
        msg_id_     = other.msg_id_;
        msg_status_ = other.msg_status_;
        msg_tensor_ = other.msg_tensor_;

        return *this;
    }

    int      msg_id_;
    int      msg_status_;
    Tensor*  msg_tensor_;

}; //Message

/**
 * @brief The GraphNode object is the base class for graph's nodes.
 * 
 * its default ctor is disabled, it can only constructed through other parameter ctor
 * 
 * it just hold the pointers of solver_, out_tensors_, it cannot create&release them 
 * 
 * it just a abstract class, user should implement PutMsg(int port_idx, const Message& msg)
 * 
 */
class GraphNode
{
public:
    static const unsigned int kForwardAll  = 0xFFFFFFFF;
    static const unsigned int kForwardStop = 0x0;

public:
    /**
     * @brief Construct from parameter
     *
     * @param node_name node's name.
     * 
     * @param solver node's solve, which will execute the computation.
     * 
     * @param out_tensors array of output tensors for solver
     * 
     * @exceptsafe No throw.
     */
    GraphNode(std::string const&           node_name,
              Solver*                      solver, 
              std::vector<Tensor*> const&  out_tensors) noexcept : node_name_(node_name), 
                                                                   solver_(solver),
                                                                   out_tensors_(out_tensors){;};

    /**
     * @brief Copy constructor
     *
     * @param other The other GraphNode object.
     * 
     * @exceptsafe No throw.
     */
    GraphNode(GraphNode const& other) noexcept : node_name_(other.node_name_), 
                                                 solver_(other.solver_),
                                                 out_tensors_(other.out_tensors_){;};

    /**
     * @brief assignment constructor
     * 
     * @param other The other GraphNode object.
     * 
     * @exceptsafe No throw.
     */
    GraphNode& operator=(GraphNode const& other) noexcept
    { 
        if(&other == this)
            return *this;
        
        node_name_   = other.node_name_;
        solver_      = other.solver_;
        out_tensors_ = other.out_tensors_;
        return *this ;
    } ;

    /**
     * @brief deconstructor
     * 
     * @exceptsafe No throw.
     */
    virtual ~GraphNode() noexcept {;};


    /**
     * @brief put msg for next node
     *
     * @param msg message will be transfered
     * 
     * it is a pure-virtual function, derived class should implement it
     * 
     * @exceptsafe No throw.
     */
    virtual bool                   PutMsg(int port_idx, const Message& msg) = 0;

    /**
     * @brief get itself index in a graph
     * 
     * @exceptsafe No throw.
     */
    std::string const&             GetNodeName() const noexcept                  { return node_name_ ; } ;

    /**
     * @brief get its solver
     * 
     * @exceptsafe No throw.
     */
    Solver*                        GetSolver() noexcept                          { return solver_; };

    /**
     * @brief get its solver through const style
     * 
     * @exceptsafe No throw.
     */
    const Solver*                  GetSolver() const noexcept                    { return solver_; };

    /**
     * @brief get its out-tensors's reference
     * 
     * @exceptsafe No throw.
     */
    std::vector<Tensor*>&          GetOutTensors() noexcept                      { return out_tensors_; } ;

    /**
     * @brief get its out-tensors's reference with const style
     * 
     * @exceptsafe No throw.
     */
    std::vector<Tensor*> const&    GetOutTensors() const noexcept                { return out_tensors_; } ;

    /**
     * @brief get flag for passed
     * 
     * @note if the node is executed after Graph start every time, this flag will be true, other will be false
     * 
     * @exceptsafe No throw.
     */
    bool                           IsPassed() const noexcept                     { return false; } ;

protected:
    std::string                    node_name_;
    Solver*                        solver_;
    std::vector<Tensor*>           out_tensors_;
}; //GraphNode

}//namespace vision_graph

#endif