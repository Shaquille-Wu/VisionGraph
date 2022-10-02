/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file graph_func_node.h
 * @brief This header file defines graph node.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef FUNC_NODE_H_
#define FUNC_NODE_H_

#include "graph_node.h"
#include "tbb/flow_graph.h"
#include <tuple>
#include <iostream>
#include <assert.h>
#include <typeinfo>
#include <cxxabi.h>
#include <logging.h>
#include <chrono> 

namespace vision_graph
{

typedef Message                                                     MultiMsg1D;
typedef std::tuple<Message, Message>                                MultiMsg2D;
typedef std::tuple<Message, Message, Message>                       MultiMsg3D;
typedef std::tuple<Message, Message, Message, Message>              MultiMsg4D;
typedef std::tuple<Message, Message, Message, Message, 
                   Message>                                         MultiMsg5D;
typedef std::tuple<Message, Message, Message, Message, 
                   Message, Message>                                MultiMsg6D;
typedef std::tuple<Message, Message, Message, Message, 
                   Message, Message, Message>                       MultiMsg7D;
typedef std::tuple<Message, Message, Message, Message, 
                   Message, Message, Message, Message>              MultiMsg8D;

typedef tbb::flow::join_node<MultiMsg2D, tbb::flow::queueing>       JoinNode2D;
typedef tbb::flow::join_node<MultiMsg3D, tbb::flow::queueing>       JoinNode3D;
typedef tbb::flow::join_node<MultiMsg4D, tbb::flow::queueing>       JoinNode4D;
typedef tbb::flow::join_node<MultiMsg5D, tbb::flow::queueing>       JoinNode5D;
typedef tbb::flow::join_node<MultiMsg6D, tbb::flow::queueing>       JoinNode6D;
typedef tbb::flow::join_node<MultiMsg7D, tbb::flow::queueing>       JoinNode7D;
typedef tbb::flow::join_node<MultiMsg8D, tbb::flow::queueing>       JoinNode8D;

typedef tbb::flow::function_node<Message, Message>                  FuncNode;
typedef FuncNode                                                    FuncNode1P1P;
typedef tbb::flow::function_node<Message>                           FuncInport1D;
typedef tbb::flow::function_node<MultiMsg2D>                        FuncInport2D;
typedef tbb::flow::function_node<MultiMsg3D>                        FuncInport3D;
typedef tbb::flow::function_node<MultiMsg4D>                        FuncInport4D;
typedef tbb::flow::function_node<MultiMsg5D>                        FuncInport5D;
typedef tbb::flow::function_node<MultiMsg6D>                        FuncInport6D;
typedef tbb::flow::function_node<MultiMsg7D>                        FuncInport7D;
typedef tbb::flow::function_node<MultiMsg8D>                        FuncInport8D;
typedef tbb::flow::broadcast_node<Message>                          FuncOutport;


typedef tbb::detail::d1::queueing_port<Message>                     JoinNodeInPort;

constexpr int kInPortMaxCount            = std::tuple_size<MultiMsg8D>::value;

class GraphFuncNode : public GraphNode
{
public:
    GraphFuncNode(std::string const&           node_name,
                  int                          enable_ports,
                  Solver*                      solver, 
                  std::vector<Tensor*> const&  out_tensors) noexcept : GraphNode(node_name, 
                                                                                 solver, 
                                                                                 out_tensors), 
                                                                                 enable_ports_(enable_ports),
                                                                                 is_branch_(false),
                                                                                 passed_(false),
                                                                                 solve_cost_(0LL),
                                                                                 is_checking_(false) {;};
    GraphFuncNode(GraphFuncNode const& other) noexcept : GraphNode(other), 
                                                         enable_ports_(other.enable_ports_),
                                                         is_branch_(other.is_branch_),
                                                         passed_(other.passed_),
                                                         solve_cost_(other.solve_cost_),
                                                         is_checking_(other.is_checking_) {;};

    GraphFuncNode& operator=(GraphFuncNode const& other) noexcept
    { 
        if(&other == this)
            return *this;
        
        node_name_     = other.node_name_;
        solver_        = other.solver_;
        out_tensors_   = other.out_tensors_;
        enable_ports_  = other.enable_ports_;
        is_branch_     = other.is_branch_;
        passed_        = other.passed_;
        solve_cost_    = other.solve_cost_;
        is_checking_   = other.is_checking_;
        return *this ;
    } ;
    virtual ~GraphFuncNode() noexcept {;};

    virtual bool                   Link(int self_output_idx, GraphFuncNode* successor, int successor_inport_idx) = 0;
    virtual int                    GetInPortCount() const noexcept = 0 ;
    virtual int                    GetOutPortCount() const noexcept = 0 ;
    virtual FuncNode*              GetReceiveNode() = 0;
    virtual FuncInport1D*          GetReceive1DPort() = 0;
    virtual JoinNodeInPort*        GetReceivePort(int port_idx) = 0;

    virtual std::vector<Tensor*>   GetRealInputTensors(std::vector<Tensor*> const& tensor_vec) const noexcept
    {
        int                   real_tensors_count = (int)(tensor_vec.size() - enable_ports_);
        std::vector<Tensor*>  real_tensors(real_tensors_count);
        for(int i = 0 ; i < real_tensors_count ; i ++)
        {
            if(kTensorReference == tensor_vec[i]->GetType())
                real_tensors[i] = (dynamic_cast<TensorReference const*>(tensor_vec[i]))->reference_;
            else
                real_tensors[i] = tensor_vec[i];
        }
        return real_tensors;
    };

    bool                           IsBranch() const noexcept                  { return is_branch_; };
    void                           SetBranch(bool flag) noexcept              { is_branch_ = flag; };

    bool                           IsPassed() const noexcept                  { return passed_ ; } ;
    void                           SetPassed(bool flag) noexcept              { passed_ = flag ; } ;

    long long int                  GetSolveCost() const noexcept               { return solve_cost_; } ;
    void                           SetSolveCost(long long int cost) noexcept   { solve_cost_ = cost ; } ;

    bool                           IsChecking() const noexcept                 { return is_checking_; };
    void                           SetChecking(bool flag) noexcept             { is_checking_ = flag; };

    virtual int                    ProcMsg(Message const** msg, int msg_count) noexcept
    {
        if(true == is_checking_)   return 0;

        unsigned int   forward_mask   = kForwardAll;
        int            res            = 0;
        Tensor*        src_tensors[kInPortMaxCount];
        int            i              = 0 ;  
        memset(src_tensors, 0,  kInPortMaxCount * sizeof(Tensor*));
        for(i = 0 ; i < msg_count ; i ++)
            src_tensors[i] = (*(msg[i])).msg_tensor_;
        std::vector<Tensor*>   in_tensors;
        transform_input_tensor(src_tensors, msg_count, in_tensors);
        auto solve_start    = std::chrono::system_clock::now();
        res = solver_->Solve(in_tensors, out_tensors_);
        auto solve_end      = std::chrono::system_clock::now();
        auto solve_cost     = std::chrono::duration_cast<std::chrono::microseconds>(solve_end - solve_start);
        solve_cost_         = (long long int)(solve_cost.count());
        if(true == is_branch_)
        {
            forward_mask = 0;
            for(i = 0 ; i < (int)(out_tensors_.size()); i ++)
                forward_mask |= ((dynamic_cast<TensorUInt32*>(out_tensors_[i]))->value_) << i;
        }

        Forward(forward_mask, res);
        passed_ = true;
        return res;
    }

    virtual void                   Forward(unsigned int mask, int res) noexcept   {;};

    virtual unsigned int           GetQueMsg(std::vector<Message>& msgs) noexcept  { msgs.clear(); return false; };

    virtual void                   ClearMsgQue(int port_idx) noexcept             {;};

    static bool                    GetForwardFlag(unsigned int forward_mask, int bit) noexcept
    {
        return (0 == ((1U << bit) & forward_mask) ? false : true);
    }

protected:
    virtual void                    transform_input_tensor(Tensor**              src, 
                                                           int                   src_count, 
                                                           std::vector<Tensor*>& dst) const noexcept
    {
        int dst_count = src_count - enable_ports_;
        dst.resize(dst_count);
        for(int i = 0 ; i < dst_count ; i ++)
        {
            if(kTensorReference == src[i]->GetType())
                dst[i] = (dynamic_cast<TensorReference*>(src[i]))->reference_;
            else
                dst[i] = src[i];
        }
    };


protected:
    int                             enable_ports_;
    bool                            is_branch_;
    bool                            passed_;
    long long int                   solve_cost_;
    bool                            is_checking_;
};


//TODO, remove GraphFuncNode1P1P temporarily
//we do not know where is the effecient balance point, so, just comment it
#if 0
class GraphFuncNode1P1P : public GraphFuncNode, public FuncNode1P1P 
{
public:
    class FuncBody
    {
    public:
        FuncBody(GraphFuncNode1P1P* func_node) noexcept : func_node_(func_node){;};
        FuncBody(FuncBody const& other) noexcept : func_node_(other.func_node_){;};
        FuncBody& operator=(FuncBody const& other)
        {
            func_node_  = other.func_node_;
            return *this;
        };
        virtual ~FuncBody() noexcept {;};

        Message operator()(Message const& in)
        {
            Message msg;
            func_node_->ProcMsg(in, msg);
            return msg;
        }

        GraphFuncNode1P1P*    func_node_;
    };
public:    
    GraphFuncNode1P1P(tbb::flow::graph&            g, 
                      int                          concurrency, 
                      std::string const&           node_name,
                      int                          enable_ports,
                      Solver*                      solver, 
                      std::vector<Tensor*> const&  out_tensors) noexcept:GraphFuncNode(node_name, enable_ports, solver, out_tensors), 
                                                                         FuncNode1P1P(g, 
                                                                                      concurrency, 
                                                                                      FuncBody(this)){;};

    virtual ~GraphFuncNode1P1P() noexcept {;};

    virtual bool             Link(int self_output_idx, GraphFuncNode* successor, int successor_inport_idx)
    {
        if(nullptr != successor->GetReceiveNode())
            tbb::flow::make_edge(*this, *(successor->GetReceiveNode()));
        else if(nullptr != successor->GetReceive1DPort())
            tbb::flow::make_edge(*this, *(successor->GetReceive1DPort()));
        else
            tbb::flow::make_edge(*this, *(successor->GetReceivePort(successor_inport_idx)));
        return true;
    }

    virtual FuncNode*        GetReceiveNode()
    {
        return this;
    }

    virtual FuncInport1D*    GetReceive1DPort()
    {
        return nullptr;
    }

    virtual JoinNodeInPort*  GetReceivePort(int port_idx)
    {
        return nullptr;
    }

    virtual bool             PutMsg(int port_idx, const Message& msg)
    {
        (void)port_idx;
        return try_put(msg) ;
    };    

    virtual void             Forward(unsigned int mask, int res) noexcept
    {
        /*
        if(1 == mask)
            try_put(Message(0, res, out_tensors_[0])) ;
        */
    }
};
#endif

class GraphFuncNode1PXP : public GraphFuncNode
{
public:
    class FuncBody
    {
    public:
        FuncBody(GraphFuncNode1PXP* func_node) noexcept : func_node_(func_node) {;};
        FuncBody(FuncBody const& other) noexcept : func_node_(other.func_node_)  {;};
        FuncBody& operator=(FuncBody const& other)
        {
            func_node_  = other.func_node_;
            return *this;
        };
        virtual ~FuncBody() noexcept {;};

        bool operator()(const Message& in)
        {
            const Message* msg[] = { &in } ;
            func_node_->ProcMsg(msg, 1);
            return true;
        }

        GraphFuncNode1PXP*                  func_node_;
    };   

public:
    GraphFuncNode1PXP(tbb::flow::graph&            g, 
                      int                          concurrency, 
                      std::string const&           node_name,
                      int                          enable_ports,
                      int                          out_count, 
                      Solver*                      solver,
                      std::vector<Tensor*> const&  out_tensors) noexcept : GraphFuncNode(node_name, 
                                                                                         enable_ports,
                                                                                         solver, 
                                                                                         out_tensors)
    {
        std::vector<FuncOutport*> out_ports_for_body(out_count);
        out_ports_.resize(out_count);
        for(int i = 0 ; i < out_count ; i ++)
            out_ports_[i] = std::unique_ptr<FuncOutport>(new FuncOutport(g));
            
        in_port_ = std::unique_ptr<FuncInport1D>(new FuncInport1D(g, concurrency, FuncBody(this)));
    };
    virtual ~GraphFuncNode1PXP() noexcept {;};

    FuncInport1D*                               GetInPort() noexcept               { return in_port_.get(); } ;

    FuncOutport*                                GetOutPort(int out_port) noexcept  { return out_ports_[out_port].get(); } ;

    virtual bool                                PutMsg(int port_idx, Message const& in) noexcept
    {
        return in_port_->try_put(in);
    }

    virtual void                                Forward(unsigned int mask, int res) noexcept
    {
        for(size_t i = 0 ; i < out_ports_.size() ; i ++)
        {
            if(true == GetForwardFlag(mask, i))
                out_ports_[i]->try_put(Message(i, res, out_tensors_[i]));
        }
    }

    virtual bool                                Link(int self_output_idx, GraphFuncNode* successor, int successor_inport_idx)
    {
        if(nullptr != successor->GetReceiveNode())
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceiveNode()));
        else if(nullptr != successor->GetReceive1DPort())
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceive1DPort()));
        else
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceivePort(successor_inport_idx)));
        return true;
    }

    int                      GetInPortCount() const noexcept  { return 1; };
    int                      GetOutPortCount() const noexcept { return (int)(out_ports_.size()); };

    virtual FuncNode*        GetReceiveNode()
    {
        return nullptr;
    }

    virtual FuncInport1D*    GetReceive1DPort()
    {
        return in_port_.get();
    }

    virtual JoinNodeInPort*  GetReceivePort(int port_idx)
    {
        return nullptr;
    }

    std::unique_ptr<FuncInport1D>              in_port_;
    std::vector<std::unique_ptr<FuncOutport>>  out_ports_;
};

template <typename JoinNodeType, typename FuncInPortType, typename MsgXDType>
class GraphFuncNodeNPXP : public GraphFuncNode
{
public:
    class FuncBody
    {
    public:
        FuncBody(GraphFuncNodeNPXP* func_node) noexcept : func_node_(func_node) {;};
        FuncBody(FuncBody const& other) noexcept : func_node_(other.func_node_) {;};
        FuncBody& operator=(FuncBody const& other)
        {
            func_node_ = other.func_node_;
            return *this;
        };
        virtual ~FuncBody() noexcept {;};

        bool operator()(MsgXDType const& in)
        {
            Message* msg[std::tuple_size<MsgXDType>::value] = { 0 };
            GraphFuncNodeNPXP::make_msg<std::tuple_size<MsgXDType>::value>(in, msg);
            func_node_->ProcMsg((Message const**)msg, std::tuple_size<MsgXDType>::value);
            return true;
        }

        GraphFuncNodeNPXP*    func_node_;
    };

    GraphFuncNodeNPXP(tbb::flow::graph&            g, 
                      int                          concurrency, 
                      std::string const&           node_name,
                      int                          enable_ports,
                      int                          out_count, 
                      Solver*                      solver,
                      std::vector<Tensor*> const&  out_tensors) noexcept : GraphFuncNode(node_name, 
                                                                                         enable_ports, 
                                                                                         solver, 
                                                                                         out_tensors)
    {
        out_ports_.resize(out_count);
        for(int i = 0 ; i < out_count ; i ++)
        {
            out_ports_[i] = std::unique_ptr<FuncOutport>(new FuncOutport(g));
        }

        join_port_ = std::unique_ptr<JoinNodeType>(new JoinNodeType(g));
        in_port_   = std::unique_ptr<FuncInPortType>(new FuncInPortType(g, concurrency, FuncBody(this)));
        init_receivers<std::tuple_size<MsgXDType>::value>();
        tbb::flow::make_edge(*join_port_, *in_port_);
    };
    virtual ~GraphFuncNodeNPXP() noexcept {;};

    JoinNodeType*                              GetInPort() noexcept               { return join_port_.get(); } ;
    FuncOutport*                               GetOutPort(int out_port) noexcept  { return out_ports_[out_port].get(); } ;

    virtual bool                               PutMsg(int port_idx, Message const& in) noexcept
    {
        assert(port_idx < (int)(std::tuple_size<MsgXDType>::value));
        return receivers_[port_idx]->try_put(in);
    }

    virtual void                               Forward(unsigned int mask, int res) noexcept
    {
        for(size_t i = 0 ; i < out_ports_.size() ; i ++)
        {
            if(true == GetForwardFlag(mask, i))
                out_ports_[i]->try_put(Message(i, res, out_tensors_[i]));
        }
    }

    virtual bool                               Link(int self_output_idx, GraphFuncNode* successor, int successor_inport_idx)
    {
        assert(self_output_idx < (int)(out_ports_.size()));
        if(nullptr != successor->GetReceiveNode())
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceiveNode()));
        else if(nullptr != successor->GetReceive1DPort())
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceive1DPort()));
        else
            tbb::flow::make_edge(*(out_ports_[self_output_idx]), *(successor->GetReceivePort(successor_inport_idx)));
        return true;
    }

    virtual FuncNode*        GetReceiveNode()
    {
        return nullptr;
    }

    virtual FuncInport1D*    GetReceive1DPort()
    {
        return nullptr;
    }

    virtual JoinNodeInPort*  GetReceivePort(int port_idx)
    {
        assert(port_idx < (int)(receivers_.size()));
        return receivers_[port_idx];
    }

    int                      GetInPortCount() const noexcept  { return std::tuple_size<MsgXDType>::value; };
    int                      GetOutPortCount() const noexcept { return (int)(out_ports_.size()); };

    unsigned int             GetQueMsg(std::vector<Message>& msgs) noexcept 
    { 
        unsigned int mask = 0 ;
        msgs.resize(receivers_.size());
        for(size_t i = 0 ; i < receivers_.size(); i ++)
        {
            Message new_msg;
            bool res = receivers_[i]->get_item(new_msg);
            if(true == res)
            {
                msgs[i] = new_msg;
                mask    = (mask | (1 << i));
            }
        }
        
        return mask; 
    };

    void                     ClearMsgQue(int port_idx) noexcept
    {
        assert(port_idx < (int)(receivers_.size()));
        receivers_[port_idx]->reset();
    }

private:
    template<size_t join_size, typename std::enable_if<2 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<3 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<4 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
        receivers_[3] = &(tbb::flow::input_port<3>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<5 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
        receivers_[3] = &(tbb::flow::input_port<3>(*join_port_));
        receivers_[4] = &(tbb::flow::input_port<4>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<6 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
        receivers_[3] = &(tbb::flow::input_port<3>(*join_port_));
        receivers_[4] = &(tbb::flow::input_port<4>(*join_port_));
        receivers_[5] = &(tbb::flow::input_port<5>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<7 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
        receivers_[3] = &(tbb::flow::input_port<3>(*join_port_));
        receivers_[4] = &(tbb::flow::input_port<4>(*join_port_));
        receivers_[5] = &(tbb::flow::input_port<5>(*join_port_));
        receivers_[6] = &(tbb::flow::input_port<6>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<8 == join_size, void>::type* = nullptr>
    void                                      init_receivers() noexcept
    {
        receivers_.resize(join_size);
        receivers_[0] = &(tbb::flow::input_port<0>(*join_port_));
        receivers_[1] = &(tbb::flow::input_port<1>(*join_port_));
        receivers_[2] = &(tbb::flow::input_port<2>(*join_port_));
        receivers_[3] = &(tbb::flow::input_port<3>(*join_port_));
        receivers_[4] = &(tbb::flow::input_port<4>(*join_port_));
        receivers_[5] = &(tbb::flow::input_port<5>(*join_port_));
        receivers_[6] = &(tbb::flow::input_port<6>(*join_port_));
        receivers_[7] = &(tbb::flow::input_port<7>(*join_port_));
    }

    template<size_t join_size, typename std::enable_if<2 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
    }

    template<size_t join_size, typename std::enable_if<3 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
    }

    template<size_t join_size, typename std::enable_if<4 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
        msg[3] = const_cast<Message*>(&(std::get<3>(in)));
    }

    template<size_t join_size, typename std::enable_if<5 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
        msg[3] = const_cast<Message*>(&(std::get<3>(in)));
        msg[4] = const_cast<Message*>(&(std::get<4>(in)));
    }

    template<size_t join_size, typename std::enable_if<6 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
        msg[3] = const_cast<Message*>(&(std::get<3>(in)));
        msg[4] = const_cast<Message*>(&(std::get<4>(in)));
        msg[5] = const_cast<Message*>(&(std::get<5>(in)));
    }

    template<size_t join_size, typename std::enable_if<7 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
        msg[3] = const_cast<Message*>(&(std::get<3>(in)));
        msg[4] = const_cast<Message*>(&(std::get<4>(in)));
        msg[5] = const_cast<Message*>(&(std::get<5>(in)));
        msg[6] = const_cast<Message*>(&(std::get<6>(in)));
    }

    template<size_t join_size, typename std::enable_if<8 == join_size, void>::type* = nullptr>
    static void                      make_msg(MsgXDType const& in, Message* msg[]) noexcept
    {
        msg[0] = const_cast<Message*>(&(std::get<0>(in)));
        msg[1] = const_cast<Message*>(&(std::get<1>(in)));
        msg[2] = const_cast<Message*>(&(std::get<2>(in)));
        msg[3] = const_cast<Message*>(&(std::get<3>(in)));
        msg[4] = const_cast<Message*>(&(std::get<4>(in)));
        msg[5] = const_cast<Message*>(&(std::get<5>(in)));
        msg[6] = const_cast<Message*>(&(std::get<6>(in)));
        msg[7] = const_cast<Message*>(&(std::get<7>(in)));
    }
protected:
    std::unique_ptr<JoinNodeType>                 join_port_;
    std::unique_ptr<FuncInPortType>               in_port_;
    std::vector<std::unique_ptr<FuncOutport>>     out_ports_;
    std::vector<JoinNodeInPort*>                  receivers_;
};

typedef GraphFuncNodeNPXP<JoinNode2D, FuncInport2D, MultiMsg2D>   GraphFuncNode2PXP;
typedef GraphFuncNodeNPXP<JoinNode3D, FuncInport3D, MultiMsg3D>   GraphFuncNode3PXP;
typedef GraphFuncNodeNPXP<JoinNode4D, FuncInport4D, MultiMsg4D>   GraphFuncNode4PXP;
typedef GraphFuncNodeNPXP<JoinNode5D, FuncInport5D, MultiMsg5D>   GraphFuncNode5PXP;
typedef GraphFuncNodeNPXP<JoinNode6D, FuncInport6D, MultiMsg6D>   GraphFuncNode6PXP;
typedef GraphFuncNodeNPXP<JoinNode7D, FuncInport7D, MultiMsg7D>   GraphFuncNode7PXP;
typedef GraphFuncNodeNPXP<JoinNode8D, FuncInport8D, MultiMsg8D>   GraphFuncNode8PXP;

}//namespace vision_graph

#endif