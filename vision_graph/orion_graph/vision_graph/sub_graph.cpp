/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file sub_graph.cpp
 * @brief This implementation for sub graph.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-25
 */


#include "sub_graph.h"
#include <utility>
#include <logging.h>
#include <fstream>
#include <iostream>
#include <assert.h>

namespace vision_graph
{

static const char    kJsonSubGraphName[]           = "name";
static const char    kJsonNodeSet[]                = "nodes";
static const char    kJsonNodeName[]               = "name";
static const char    kJsonNodePrintProfile[]       = "print_profile";
static const char    kJsonNodeInputs[]             = "inputs";
static const char    kJsonNodeOutputs[]            = "outputs";
static const char    kJsonNodeEnablePorts[]        = "enable_ports";
static const char    kJsonInputsInit[]             = "inputs_init";
static const char    kJsonInputsInitInputIdx[]     = "input_idx";
static const char    kJsonInputsInitTensorName[]   = "init_tensor";
static const char    kJsonNodeSolver[]             = "solver";
static const char    kJsonNodeSolverName[]         = "name";
static const char    kJsonNodeSolverClassName[]    = "class";
static const char    kJsonSolverName[]             = "name";
static const char    kJsonSolverFriends[]          = "friends";

int SubGraph::ParseSubGraph(nlohmann::json const&         sub_graph_json, 
                            SolverFactory const*          solver_factory,
                            SubGraphParam&                sub_graph_param,
                            std::vector<nlohmann::json>&  solvers_json) noexcept
{
    sub_graph_param.sub_graph_name_ = sub_graph_json.at(kJsonSubGraphName);
    if(true == sub_graph_json.contains(kJsonNodePrintProfile))
        sub_graph_param.print_profile_ = sub_graph_json.at(kJsonNodePrintProfile);

    solvers_json.clear();
    int    i = 0;
    if(false == sub_graph_json.contains(kJsonNodeSet))
    {
        LOG(ERROR) << "sub_graph " << sub_graph_param.sub_graph_name_ << ", cannot found \"nodes\"" << std::endl;
        return -1;
    }
    std::vector<nlohmann::json>  nodes_json     = sub_graph_json.at(kJsonNodeSet);
    int                          json_node_size = (int)(nodes_json.size());
    std::vector<NodeParam>       vec_node_params(json_node_size);
    solvers_json.resize(json_node_size);
    std::map<std::string, int>   nodes_name_idx_map;
    for (i = 0 ; i < json_node_size; i++)
    {
        vec_node_params[i].node_name_  = nodes_json[i].at(kJsonNodeName);
        if("" == vec_node_params[i].node_name_)
        {
            LOG(ERROR) << "sub_graph " << sub_graph_param.sub_graph_name_ << ", found node" << i << " name is empty" << std::endl;
            return -2;
        }

        if(true == nodes_json[i].contains(kJsonNodeInputs))
        {
            std::vector<nlohmann::json>   input_ports_json = nodes_json[i].at(kJsonNodeInputs);
            vec_node_params[i].inputs_  = parse_node_inputs(input_ports_json);
        }

        if(true == nodes_json[i].contains(kJsonNodeOutputs))
        {
            std::vector<nlohmann::json>   output_ports_json = nodes_json[i].at(kJsonNodeOutputs);
            vec_node_params[i].outputs_ = parse_node_outputs(output_ports_json);
        }

        if(true == nodes_json[i].contains(kJsonNodeEnablePorts))
        {
            std::vector<nlohmann::json>   enable_ports_json = nodes_json[i].at(kJsonNodeEnablePorts);
            vec_node_params[i].enable_ports_ = parse_node_inputs(enable_ports_json);
        }

        if(true == nodes_json[i].contains(kJsonInputsInit))
        {
            std::vector<nlohmann::json>   inputs_init_json = nodes_json[i].at(kJsonInputsInit);
            vec_node_params[i].inputs_init_ = parse_inputs_init(inputs_init_json);
        }

        if(true == nodes_json[i].contains(kJsonNodeSolver))
        {
            nlohmann::json   solver_param_json  = nodes_json[i].at(kJsonNodeSolver);
            vec_node_params[i].solver_     = parse_node_solver(solver_param_json);
            if("" == vec_node_params[i].solver_.solver_name_)
            {
                LOG(ERROR) << "node " << vec_node_params[i].node_name_ << ", its solver_name is empty" << std::endl;
                return -3;
            }   
            if("" == vec_node_params[i].solver_.class_)
            {
                LOG(ERROR) << "node " << vec_node_params[i].node_name_ << ", its class is empty" << std::endl;
                return -4;
            }
            solvers_json[i] = solver_param_json;
        }
        else
        {
            LOG(ERROR) << "sub_graph " << sub_graph_param.sub_graph_name_ << ", cannot find solver" << std::endl;
            return -5;
        }

        std::map<std::string, int>::const_iterator  iter = nodes_name_idx_map.find(vec_node_params[i].node_name_);
        if(iter != nodes_name_idx_map.end())
        {
            int other_node_idx = iter->second;
            LOG(ERROR) << "sub_graph " << sub_graph_param.sub_graph_name_ << ", node" << i << " its name is repeat with node" 
                      << other_node_idx << "(" << vec_node_params[i].node_name_ << ")" << std::endl;
        }
        nodes_name_idx_map[vec_node_params[i].node_name_] = i;
    }

    //tensor_name, <node_idx, out_idx>
    std::map<std::string, LinkParam>  out_tensors_map;
    bool res = check_nodes_input_output(vec_node_params, out_tensors_map);
    assert(true == res);
    if(false == res)    return -6;

    res      = inference_nodes_link(vec_node_params, out_tensors_map);
    assert(true == res);
    if(false == res)    return -7;

    res      = check_sub_graph(vec_node_params, nodes_name_idx_map, solver_factory);
    assert(true == res);
    if(false == res)    return -8;

    sub_graph_param.nodes_ = vec_node_params;

    return 0;
}   

bool SubGraph::CheckSubGraphsNodes(std::vector<SubGraphParam> const& sub_graphs) noexcept
{
    int  i               = 0 ;
    int  j               = 0 ;
    int  k               = 0 ;
    int  m               = 0 ;
    int  n               = 0 ;
    int  sub_graph_count = (int)(sub_graphs.size());
    //check solvers' friends
    for(i = 0 ; i < sub_graph_count ; i ++)
    {
        const SubGraphParam&           sub_graph   = sub_graphs[i];
        const std::vector<NodeParam>&  graph_nodes = sub_graph.nodes_;
        int                            node_count  = (int)(graph_nodes.size());
        for(j = 0 ; j < node_count ; j ++)
        {
            int    friends_count = (int)(graph_nodes[j].solver_.friends_.size());
            for(k = 0 ; k < friends_count ; k ++)
            {
                bool friends_valid = false;
                std::string const&   cur_friend_name = graph_nodes[j].solver_.friends_[k];
                for(m = 0 ; m < sub_graph_count ; m ++)
                {
                    const SubGraphParam&           other_sub_graph   = sub_graphs[m];
                    const std::vector<NodeParam>&  other_graph_nodes = other_sub_graph.nodes_;
                    int                            other_node_count  = (int)(other_graph_nodes.size());
                    for(n = 0 ; n < other_node_count ; n ++)
                    {
                        std::string const&  other_solver_name = other_graph_nodes[n].solver_.solver_name_;
                        if(other_solver_name == cur_friend_name)
                        {
                            friends_valid = true;
                            break;
                        }
                    }
                    if(true == friends_valid)
                        break;
                }
                if(false == friends_valid)
                {
                    std::cout << "sub_graph(" << sub_graph.sub_graph_name_ << ")" << " node" << i << "(" << graph_nodes[i].node_name_ 
                                << ") its solver's friend(" << cur_friend_name << ") cannot be found in graph" << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
};

bool SubGraph::Build(SubGraphParam const&                                  sub_graph, 
                     std::map<std::string, Solver*>&                       name_2_solvers_map,
                     std::map<std::string, std::pair<Tensor*, Solver*> >&  tensor_outs_map) noexcept
{
    const std::vector<NodeParam>&  graph_nodes = sub_graph.nodes_;
    int                            node_count  = (int)(graph_nodes.size());
    int  i          = 0;
    int  j          = 0;
    int  k          = 0;
    bool res        = false;
    sub_graph_name_ = sub_graph.sub_graph_name_;
    print_profile_  = sub_graph.print_profile_;
    nodes_.resize(node_count);
    tensor_out_map_.clear();
    nodes_tensor_out_vec_.resize(node_count);
    solvers_.resize(node_count);
    for(i = 0 ; i < node_count ; i ++)
    {
        std::string const&                       solver_name = sub_graph.nodes_[i].solver_.solver_name_;
        std::map<std::string, Solver*>::iterator iter_solver = name_2_solvers_map.find(solver_name);
        if (name_2_solvers_map.end() == iter_solver) {
            LOG(ERROR) << "name_2_solvers_map.end() == iter_solver";
            return false;
        }

            
        solvers_[i] = iter_solver->second;
        int                   out_tensors_count = (int)(graph_nodes[i].outputs_.size());
        std::vector<Tensor*>  new_tensors(out_tensors_count);
        for(j = 0 ; j < out_tensors_count ; j ++)
        {
            std::string const& tensor_out_name = graph_nodes[i].outputs_[j].tensor_name_;
            std::map<std::string, std::pair<Tensor*, Solver*> >::iterator  iter_tensor = tensor_outs_map.find(tensor_out_name);
            if (tensor_outs_map.end() == iter_tensor) {
                LOG(ERROR) << "tensor_outs_map.end() == iter_tensor";
                
             return false;
            }
               
            std::pair<Tensor*, Solver*>  tensor_solver_pair = iter_tensor->second;
            new_tensors[j] = tensor_solver_pair.first;
            tensor_out_map_[tensor_out_name] = tensor_solver_pair;
        }
        nodes_tensor_out_vec_[i]  = std::make_pair(graph_nodes[i].node_name_, new_tensors);
    }

    //check input&output tensor for every solver
    for(i = 0 ; i < node_count ; i ++)
    {
        std::vector<Tensor*>       in_tensors(graph_nodes[i].inputs_.size());
        std::vector<std::string>   in_tensor_names(graph_nodes[i].inputs_.size());
        //we just select first input for in_tensors, we can not travel all of the possible
        for(j = 0 ; j < (int)(in_tensors.size()) ; j ++)
        {
            if(graph_nodes[i].inputs_[j].links_.size() > 0)
            {
                int node_idx   = graph_nodes[i].inputs_[j].links_[0].node_idx_;
                int output_idx = graph_nodes[i].inputs_[j].links_[0].port_idx_;
                in_tensors[j]      = (nodes_tensor_out_vec_[node_idx].second)[output_idx];
                in_tensor_names[j] = graph_nodes[node_idx].outputs_[output_idx].tensor_name_;
            }
        }
        //we should judge it during runtime, instead of initial stage, if in_tensor is a TensorReference
        if(false == contain_reference_input(in_tensors, in_tensor_names))
            res = solvers_[i]->CheckTensor(in_tensors, nodes_tensor_out_vec_[i].second);
        if(false == res)
        {
            LOG(ERROR) << "node" << i << "(" << graph_nodes[i].node_name_ << "), checktensor failed" << std::endl;
            return false;
        }
    }

    for(i = 0 ; i < node_count ; i ++)
    {
        int  enable_count       = (int)(graph_nodes[i].enable_ports_.size());
        int  input_count        = (int)(graph_nodes[i].inputs_.size());
        int  total_input_count  = enable_count + input_count;
        total_input_count       = (0 != total_input_count ?  total_input_count : 1);
        nodes_[i] = std::unique_ptr<GraphFuncNode>(create_func_node(graph_nodes[i].node_name_, 
                                                                    total_input_count,
                                                                    enable_count,
                                                                    (int)(graph_nodes[i].outputs_.size()),
                                                                    solvers_[i],
                                                                    nodes_tensor_out_vec_[i].second));
        assert(nullptr != nodes_[i]);
        if(nullptr == nodes_[i])
            return false;
        if(kBranchSolverMap.end() != kBranchSolverMap.find(graph_nodes[i].solver_.class_))
            nodes_[i]->SetBranch(true);
    }

    for(i = 0 ; i < node_count ; i ++)
    {
        GraphFuncNode*  cur_node          = nodes_[i].get();
        int             output_ports_size = (int)(graph_nodes[i].outputs_.size());
        for(j = 0 ; j < output_ports_size ; j ++)
        {
            std::vector<LinkParam> const&   cur_port_links        = graph_nodes[i].outputs_[j].links_;
            int                             cur_port_links_count  = (int)(cur_port_links.size());
            for(k = 0 ; k < cur_port_links_count ; k ++)
            {
                LinkParam const&    cur_link = cur_port_links[k];
                cur_node->Link(j, nodes_[cur_link.node_idx_].get(), cur_link.port_idx_);
            }
        }
    }
/*
    auto cur_iter = graph_.begin();
    while(cur_iter != graph_.end())
    {
        cur_iter ++;
    }
*/
    input_node_idx_  = find_input_nodes(graph_nodes);
    inputs_init_     = find_input_init_nodes(graph_nodes);
    first_start_     = true;
    nodes_param_     = sub_graph.nodes_;

    return true;
}

int SubGraph::Destroy() noexcept
{
    int i = 0 ;
    nodes_tensor_out_vec_.clear();
    tensor_out_map_.clear();
    solvers_.clear();
    for(i = 0 ; i < (int)(nodes_.size()) ; i ++)
        nodes_[i].reset(nullptr);
    nodes_.clear();
    nodes_param_.clear();

    return 0;
}

int SubGraph::Start() noexcept
{
    if(input_node_idx_.size() <= 0) return kErrCodePutMsgFailed;

    int i         = 0 ;
    int j         = 0;
    int k         = 0;
    int node_size = (int)(nodes_.size());

    for(i = 0 ; i < node_size ; i ++)
    {
        nodes_[i]->SetPassed(false);
        nodes_[i]->SetSolveCost(0LL);
    }

    if(true == first_start_)
    {
        int  inputs_init_count = (int)(inputs_init_.size());
        for(i = 0 ; i < inputs_init_count ; i ++)
        {
            int node_idx = inputs_init_[i].first;
            for(j = 0 ; j < (int)(inputs_init_[i].second.size()) ; j ++)
            {
                int                input_idx   = (inputs_init_[i].second)[j].input_idx_;
                std::string const& tensor_name = (inputs_init_[i].second)[j].init_tensor_name_;
                std::map<std::string, std::pair<Tensor*, Solver*> >::const_iterator iter = tensor_out_map_.find(tensor_name);
                if(tensor_out_map_.end() == iter)
                {
                    LOG(ERROR) << "node" << node_idx << "(" << nodes_[node_idx]->GetNodeName() << "), cannot find init_tensor" << std::endl;
                    return false;
                }
                Tensor* tensor = (iter->second).first;
                nodes_[node_idx]->PutMsg(input_idx, Message(0, 0, tensor));
            }
        }
        first_start_ = false;
    }

    Message  msg;
    bool     res = false;
    for(auto input_idx : input_node_idx_)
    {
        msg.msg_id_     = 0;
        msg.msg_status_ = 0;
        msg.msg_tensor_ = (nodes_tensor_out_vec_[input_idx].second)[0];
        res = nodes_[input_idx]->PutMsg(0, msg);
    }

    return (true == res ? 0 : kErrCodePutMsgFailed);
}

void SubGraph::Wait() noexcept
{
    graph_.wait_for_all();
    clear_msg_que();
    int i         = 0 ;
    int node_size = (int)(nodes_.size());
    if(print_profile_)
    {
        for(i = 0 ; i < node_size ; i ++)
        {
            long long int solve_cost = nodes_[i]->GetSolveCost();
            LOG(INFO) << "node: " << nodes_[i]->GetNodeName() << ", passed: " << nodes_[i]->IsPassed() << " cost: " << solve_cost;
        }
    }
};

std::vector<InPortParam> SubGraph::parse_node_inputs(std::vector<nlohmann::json> const& inputs_json) noexcept
{
    int                        i                  = 0;
    int                        j                  = 0;
    int                        intput_ports_count = (int)(inputs_json.size());
    std::vector<InPortParam>   intput_ports(intput_ports_count);
    for(i = 0 ; i < intput_ports_count ; i ++)
    {
        std::vector<nlohmann::json>  links_json  = inputs_json[i];
        int                          links_count = (int)(links_json.size());
        intput_ports[i].tensor_names_.resize(links_count);
        for(j = 0 ; j < links_count ; j ++)
            intput_ports[i].tensor_names_[j] = links_json[j].get<std::string>();
    }

    return intput_ports;
};

std::vector<OutPortParam> SubGraph::parse_node_outputs(std::vector<nlohmann::json> const& outputs_json) noexcept
{
    int                         i                  = 0;
    int                         output_ports_size  = (int)(outputs_json.size());
    std::vector<OutPortParam>   output_ports(output_ports_size);
    int                         output_total       = 0;
    for(i = 0 ; i < output_ports_size ; i ++)
        output_ports[i].tensor_name_ = outputs_json[i].get<std::string>();

    return output_ports;
};

std::vector<InputInitParam> SubGraph::parse_inputs_init(std::vector<nlohmann::json> const& inputs_init_json) noexcept
{
    int                           i                  = 0;
    int                           inputs_init_count  = (int)(inputs_init_json.size());
    std::vector<InputInitParam>   inputs_init(inputs_init_count);
    for(i = 0 ; i < inputs_init_count ; i ++)
    {
        inputs_init[i].input_idx_        = inputs_init_json[i].at(kJsonInputsInitInputIdx);
        inputs_init[i].init_tensor_name_ = inputs_init_json[i].at(kJsonInputsInitTensorName).get<std::string>();
    }
    return inputs_init;
}

SolverParam SubGraph::parse_node_solver(nlohmann::json const& solver_json) noexcept
{
    SolverParam solver;
    solver.solver_name_   = solver_json.at(kJsonNodeSolverName);
    solver.class_         = solver_json.at(kJsonNodeSolverClassName);
    if(solver_json.contains(kJsonSolverFriends))
    {
        std::vector<nlohmann::json> friends_json = solver_json.at(kJsonSolverFriends);
        int                         friends_cnt  = (int)(friends_json.size());
        int                         i            = 0;
        solver.friends_.resize(friends_cnt);
        for(i = 0 ; i < friends_cnt ; i ++)
            solver.friends_[i] = friends_json[i].get<std::string>();
    }

    return solver;
};

bool SubGraph::check_nodes_input_output(std::vector<NodeParam> const&       graph_nodes,
                                        std::map<std::string, LinkParam>&   out_tensors_map) noexcept
{
    int i = 0, j = 0, k = 0, m = 0 ;
    int nodes_count = (int)(graph_nodes.size());

    //1. first, we check nodes' name, empty, and repeat
    for(i = 0 ; i < nodes_count ; i ++)
    {
        if("" == graph_nodes[i].node_name_)
        {
            LOG(ERROR) << "node " << i << ",  name is empty, " << std::endl;
                  


            return false;
        }

        for(j = 0 ; j < nodes_count ; j ++)
        {
            if(i != j)
            {
                if(graph_nodes[i].node_name_ == graph_nodes[j].node_name_)
                {
                            


                    LOG(ERROR) << "node name is repeat, " << i << " and " << j << std::endl;
                    return false;
                }
            }
        }
    }

    //2. second, we check nodes' output, empty or repeat
    for(i = 0 ; i < nodes_count ; i ++)
    {
        int cur_outport_count = (int)(graph_nodes[i].outputs_.size());
        if(0 == cur_outport_count && kZeroOutputSolverMap.end() == kZeroOutputSolverMap.find(graph_nodes[i].solver_.class_))
        {
            LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ 
                      << ", its output should not be zero" << std::endl;
                          


            return false;
        }

        for(j = 0 ; j < cur_outport_count ; j ++)
        {
            if("" == graph_nodes[i].outputs_[j].tensor_name_)
            {
                LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", output(" << j 
                          << "), is empty" << std::endl;
                              


                return false;
            }
            std::map<std::string, LinkParam>::const_iterator  iter_tensor = out_tensors_map.find(graph_nodes[i].outputs_[j].tensor_name_);
            if(iter_tensor != out_tensors_map.end())
            {
                int  repeat_node_idx = (iter_tensor->second).node_idx_;
                int  repeat_out_idx  = (iter_tensor->second).port_idx_;
                if(repeat_node_idx == i)
                {
                    LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", output(" << j << ", " << repeat_out_idx 
                              << "), their names are repeat" << std::endl;
                                    


                }
                else
                {
                    LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", output(" << j << "), and node "
                                << repeat_node_idx << graph_nodes[k].node_name_ << ", output(" 
                                << repeat_out_idx << "), their names are repeat" << std::endl;
                    


                }
                return false;
            }
            out_tensors_map[graph_nodes[i].outputs_[j].tensor_name_] = LinkParam(i, j);
        }
    }

    //3. third, we check nodes' input, empty, or repeat in group
    bool res = check_node_input(graph_nodes, out_tensors_map, false);

    if (false == res) {
                  
                        return false;
    
    }
        

    //3. third, we check nodes' enable_port, empty, or repeat in group
    res = check_node_input(graph_nodes, out_tensors_map, true);
    if (false == res) {
                 

    return false;
    }
        

    //4. forth, check count(input+enable_ports) is valid
    for(i = 0 ; i < nodes_count ; i ++)
    {
        int input_total = (int)(graph_nodes[i].inputs_.size() + graph_nodes[i].enable_ports_.size());
        if(input_total > kInPortMaxCount)
        {
            LOG(ERROR) << "node" << i << "(" << graph_nodes[i].node_name_ << "), its (in + enable_ports) is " << input_total 
                      << ", it beyond " << kInPortMaxCount << std::endl;
        }
    }

    //5. fifth, check input_init is valid
    for(i = 0 ; i < nodes_count ; i ++)
    {
        int    inputs_count      = (int)(graph_nodes[i].inputs_.size());
        int    inputs_init_count = (int)(graph_nodes[i].inputs_init_.size());
        for(j = 0 ; j < inputs_init_count ; j ++)
        {
            if(graph_nodes[i].inputs_init_[j].input_idx_ >= inputs_count)
            {
                LOG(ERROR) << "node" << i << "(" << graph_nodes[i].node_name_ << "), its input_init(" << j << ") is " 
                          << graph_nodes[i].inputs_init_[j].input_idx_ << ", is >= " << inputs_count << std::endl;
                       
                
                return false;
            }
            std::string const&  init_tensor_name_ = graph_nodes[i].inputs_init_[j].init_tensor_name_;
            if("" == init_tensor_name_)
            {
                LOG(ERROR) << "node" << i << "(" << graph_nodes[i].node_name_ << "), its input_init(" << j 
                          << ") its tensor_name is empty" << std::endl;
                       
                
                return false;
            }
            if(out_tensors_map.find(init_tensor_name_) == out_tensors_map.end())
            {
                LOG(ERROR) << "node" << i << "(" << graph_nodes[i].node_name_ << "), its input_init(" << j << ") its tensor_name is " 
                          << init_tensor_name_ << ", cannot be found in out_tensors" << std::endl;
                      
                
                return false;
            }
        }
    }

    return true;
}

bool SubGraph::check_node_input(std::vector<NodeParam> const&              graph_nodes,
                                std::map<std::string, LinkParam> const&    out_tensors_map,
                                bool                                       is_enable_ports) noexcept
{
    int i = 0, j = 0, k = 0, m = 0 ;
    int nodes_count = (int)(graph_nodes.size());
    std::string   print_keyname = (is_enable_ports ? "enable_port" : "input");
    for(i = 0 ; i < nodes_count ; i ++)
    {
        int  outport_count = (int)(graph_nodes[i].outputs_.size());
        std::vector<InPortParam> const*  input_ports = nullptr;
        if(true == is_enable_ports)
            input_ports = &(graph_nodes[i].enable_ports_);
        else
            input_ports = &(graph_nodes[i].inputs_);
        int  inports_count       = (int)(input_ports->size());
        int  inports_total_count = (int)(graph_nodes[i].enable_ports_.size() + graph_nodes[i].inputs_.size());
        if(false == is_enable_ports && 0 == inports_total_count && kInputSolverMap.end() == kInputSolverMap.find(graph_nodes[i].solver_.class_))
        {
                        
            
             LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", its input should not be zero" << std::endl;
            return false;
        }

        if(kInputSolverMap.end() != kInputSolverMap.find(graph_nodes[i].solver_.class_))
        {
            if(input_ports->size() > 0)
            {
                std::cout << "node " << i << ", " << graph_nodes[i].node_name_ << ", it is a data_node, its inputs(enable_ports including) should be zero" << std::endl;
                return false;
            }
        }

        for(j = 0 ; j < inports_count ; j ++)
        {
            int  links_count = (int)((*input_ports)[j].tensor_names_.size());
            if(0 == links_count)
            {
                    

                LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", " << print_keyname << j << ", its links is zero" << std::endl;
                return false;
            }
            for(k = 0 ; k < links_count ; k ++)
            {
                if("" == (*input_ports)[j].tensor_names_[k])
                {
                  

                    LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", " << print_keyname << j << ", links" 
                              << k << ", its name is empty" << std::endl;
                    return false;
                }
                std::map<std::string, LinkParam>::const_iterator  iter_tensor = out_tensors_map.find((*input_ports)[j].tensor_names_[k]);
                if(iter_tensor == out_tensors_map.end())
                {
                       

                    LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", " << print_keyname << j << ", links" 
                              << k << ", its name cannot be found in out_tensors" << std::endl;
                    return false;
                }
                for(m = 0 ; m < links_count ; m ++)
                {
                    if(m != k)
                    {
                        if((*input_ports)[j].tensor_names_[k] == (*input_ports)[j].tensor_names_[m])
                        {
                  

                            LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", " << print_keyname << j << ", links" 
                                    << k << " and links " << m << ", name " << (*input_ports)[j].tensor_names_[k] 
                                    << " they are repeat" << std::endl;
                            return false;
                        }
                    }
                }
                for(m = 0 ; m < outport_count ; m ++)
                {
                    if((*input_ports)[j].tensor_names_[k] == graph_nodes[i].outputs_[m].tensor_name_)
                    {
                

                        LOG(ERROR) << "node " << i << ", " << graph_nodes[i].node_name_ << ", " << print_keyname << j << ", links" 
                                << k << ", " << (*input_ports)[j].tensor_names_[k] << ", output" << m
                                << ", they are repeat, we cannot link to itself" << std::endl;
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

bool SubGraph::inference_nodes_link(std::vector<NodeParam>&                  graph_nodes, 
                                    std::map<std::string, LinkParam> const&  out_tensors_map) noexcept
{
    int i = 0, j = 0, k = 0, m = 0 ;
    int nodes_count = (int)(graph_nodes.size());

    std::map<std::string, std::vector<LinkParam> >  out_tensor_link_map;
    for(i = 0 ; i < nodes_count ; i ++)
    {
        int input_ports_count  = (int)(graph_nodes[i].inputs_.size());
        for(j = 0 ; j < input_ports_count ; j ++)
        {
            int  ports_links_count = (int)(graph_nodes[i].inputs_[j].tensor_names_.size());
            graph_nodes[i].inputs_[j].links_.resize(ports_links_count);
            for(k = 0 ; k < ports_links_count ; k ++)
            {
                std::string const&  in_tensor_name   = graph_nodes[i].inputs_[j].tensor_names_[k];
                LinkParam const&    out_tensor_node  = (out_tensors_map.find(in_tensor_name))->second;
                graph_nodes[i].inputs_[j].links_[k] = out_tensor_node;
                std::map<std::string, std::vector<LinkParam> >::iterator   iter     = out_tensor_link_map.find(in_tensor_name);
                if(iter == out_tensor_link_map.end())
                    out_tensor_link_map[in_tensor_name] = std::vector<LinkParam>({ LinkParam(i, j) });
                else
                    (iter->second).push_back(LinkParam(i, j));
            }
        }
        
        int   enable_ports_count  = (int)(graph_nodes[i].enable_ports_.size());
        for(j = 0 ; j < enable_ports_count ; j ++)
        {
            int  ports_links_count = (int)(graph_nodes[i].enable_ports_[j].tensor_names_.size());
            graph_nodes[i].enable_ports_[j].links_.resize(ports_links_count);
            for(k = 0 ; k < ports_links_count ; k ++)
            {
                std::string const&  in_tensor_name   = graph_nodes[i].enable_ports_[j].tensor_names_[k];
                LinkParam const&    out_tensor_node  = (out_tensors_map.find(in_tensor_name))->second;
                graph_nodes[i].enable_ports_[j].links_[k] = out_tensor_node;
                std::map<std::string, std::vector<LinkParam> >::iterator   iter     = out_tensor_link_map.find(in_tensor_name);
                int                                                        port_idx = input_ports_count + j;
                if(iter == out_tensor_link_map.end())
                    out_tensor_link_map[in_tensor_name] = std::vector<LinkParam>({ LinkParam(i, port_idx) });
                else
                    (iter->second).push_back(LinkParam(i, port_idx));
            }
        }
    }

    for(i = 0 ; i < nodes_count ; i ++)
    {
        int  outports_count = (int)(graph_nodes[i].outputs_.size());
        for(j = 0 ; j < outports_count ; j ++)
        {
            const std::string&                                         tensor_name = graph_nodes[i].outputs_[j].tensor_name_;
            std::map<std::string, std::vector<LinkParam> >::iterator   iter        = out_tensor_link_map.find(tensor_name);
            if(iter != out_tensor_link_map.end())
                graph_nodes[i].outputs_[j].links_ = (iter->second);
        }
    }

    return true;
}

bool SubGraph::check_sub_graph(std::vector<NodeParam> const&        graph_nodes, 
                               std::map<std::string, int> const&    nodes_name_idx_map, 
                               SolverFactory const*                 solver_factory) noexcept
{
    int    i          = 0;
    int    j          = 0;
    int    k          = 0;
    int    m          = 0;
    int    nodes_size = (int)(graph_nodes.size());

    //1. solver can be found in solver_factory?
    if(nullptr != solver_factory)
    {
        for(i = 0 ; i < nodes_size ; i ++)
        {
            if("" == graph_nodes[i].solver_.class_)
            {
                std::cout << "node " << i << ", " << graph_nodes[i].node_name_ << ", solver name is empty" << std::endl;
                return false;
            }

            if(false == solver_factory->Check(graph_nodes[i].solver_.class_))
            {
                std::cout << "node" << i << ", " << graph_nodes[i].node_name_ << ", solver(" << graph_nodes[i].solver_.class_ 
                        << ") cannot be found in factory" << std::endl;
                return false;
            }
        }
    }

    //2.data node(solver class_ is SolverData) only 1 ?
    int  program_input_count = 0;
    for(i = 0 ; i < nodes_size ; i ++)
    {
        if(0 == graph_nodes[i].inputs_.size())
        {
            if(kInputSolverMap.end() != kInputSolverMap.find(graph_nodes[i].solver_.class_))
            {
                if(graph_nodes[i].outputs_.size() <= 0)
                {
                    LOG(ERROR) << "node" << i << ", " << graph_nodes[i].node_name_ << ", its a data node, but its output is 0" << std::endl;
                    return false;
                }

                program_input_count ++;
            }
            else
            {
                if(0 == graph_nodes[i].enable_ports_.size())
                {
                    LOG(ERROR) << "node" << i << ", " << graph_nodes[i].node_name_ << ", its not a data node, but its input is 0" << std::endl;
                    return false;
                }
            }
        }
        else
        {
            if(kInputSolverMap.end() != kInputSolverMap.find(graph_nodes[i].solver_.class_))
            {
                LOG(ERROR) << "node" << i << ", " << graph_nodes[i].node_name_ 
                          << ", it is a data node, but its input is not 0" << std::endl;
                return false;
            }
        }
    }

    if(0 == program_input_count)
    {
        LOG(ERROR) << "cannot find input node, its solver's class should be :" << std::endl;
        for(auto str : kInputSolverMap)
            LOG(ERROR) << str.first << std::endl;
        return false;
    }

    return true;
}; 

bool SubGraph::contain_reference_input(std::vector<Tensor*> const&     inputs, 
                                       std::vector<std::string> const& input_names) noexcept
{
    int    i = 0, j = 0, k = 0;
    int    input_count    = (int)(inputs.size());
    for(i = 0 ; i < input_count ; i ++)
    {
        const Tensor*       input_tensor      = inputs[i];
        const std::string&  input_name        = input_names[i];
        if(nullptr != input_tensor && kTensorReference == input_tensor->GetType())
            return true;
    }

    return false;
};

GraphFuncNode* SubGraph::create_func_node(std::string const&           node_name,
                                          int                          total_in_ports, 
                                          int                          enable_ports,
                                          int                          out_ports, 
                                          Solver*                      solver,
                                          std::vector<Tensor*> const&  out_tensors) noexcept
{
    GraphFuncNode* new_node = nullptr;
    switch(total_in_ports)
    {
        case 1:
            new_node = new GraphFuncNode1PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 2:
            new_node = new GraphFuncNode2PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 3:
            new_node = new GraphFuncNode3PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 4:
            new_node = new GraphFuncNode4PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 5:
            new_node = new GraphFuncNode5PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 6:
            new_node = new GraphFuncNode6PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 7:
            new_node = new GraphFuncNode7PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        case 8:
            new_node = new GraphFuncNode8PXP(graph_, 1, node_name, enable_ports, out_ports, solver, out_tensors);
            break;
        default:
            break;
    }
    return new_node;
}

std::vector<int> SubGraph::find_input_nodes(std::vector<NodeParam> const&  graph_nodes) noexcept
{
    int                i              = 0;
    int                j              = 0;
    int                nodes_size     = (int)(graph_nodes.size());
    std::vector<int>   input_node_idx(0);
    for(i = 0 ; i < nodes_size ; i ++)
    {
        if(kInputSolverMap.end() != kInputSolverMap.find(graph_nodes[i].solver_.class_))
        {
            input_node_idx.push_back(i);
        }
    }

    return input_node_idx;
};

std::vector<std::pair<int, std::vector<InputInitParam> > > SubGraph::find_input_init_nodes(std::vector<NodeParam> const&  graph_nodes) noexcept
{
    int                i              = 0;
    int                j              = 0;
    int                nodes_size     = (int)(graph_nodes.size());
    std::vector<std::pair<int, std::vector<InputInitParam> > >   inputs_init(0);
    for(i = 0 ; i < nodes_size ; i ++)
    {
        int inputs_init_count = (int)(graph_nodes[i].inputs_init_.size());
        if(inputs_init_count > 0)
        {
            std::vector<InputInitParam>  init_tensor = graph_nodes[i].inputs_init_;
            inputs_init.push_back(std::make_pair(i, init_tensor));
        }
    }

    return inputs_init;
};

int SubGraph::clear_msg_que() noexcept
{
    int                                 i                     = 0 ;
    int                                 j                     = 0 ;
    int                                 k                     = 0 ;
    int                                 node_count            = (int)(nodes_.size());
    int                                 input_init_node_count = (int)(inputs_init_.size());
    std::vector<std::vector<std::tuple<int, bool, Message> > >  inputs_init_msg(input_init_node_count);
    std::vector<std::vector<Message> >  node_msgs(node_count);
    std::vector<unsigned int>           vec_mask(node_count);
    int                                 remain_count          = 0;
    for(i = 0 ; i < node_count ; i ++)
    {
        nodes_[i]->SetChecking(true);
        node_msgs[i].clear();
    }

    //save the loop_init msg
    for(i = 0 ; i < input_init_node_count ; i ++)
    {
        int                                 node_idx          = inputs_init_[i].first;
        std::vector<InputInitParam> const&  inputs_init_param = inputs_init_[i].second;
        int                                 inputs_init_count = (int)(inputs_init_param.size());
        inputs_init_msg[i].resize(inputs_init_count);
        for(j = 0 ; j < inputs_init_count; j ++)
        {
            std::get<0>(inputs_init_msg[i][j]) = inputs_init_param[j].input_idx_;
            std::get<1>(inputs_init_msg[i][j]) = false;
        }
        vec_mask[node_idx]  = nodes_[node_idx]->GetQueMsg(node_msgs[node_idx]);
        int    port_count   = (int)(nodes_[node_idx]->GetInPortCount());
        for(j = 0 ; j < port_count ; j ++)
        {
           for(k = 0 ; k < inputs_init_count ; k ++)
           {
               if(j == std::get<0>(inputs_init_msg[i][k]))
               {
                    if(((1U << j) & vec_mask[node_idx]) != 0)
                    {
                        std::get<1>(inputs_init_msg[i][k]) = true;
                        std::get<2>(inputs_init_msg[i][k]) = node_msgs[node_idx][j];
                    }
               }
           }
        }
    }
        
    //pump out all of message
    while (true)
    {
        remain_count = 0 ;
        for(i = 0 ; i < node_count ; i ++)
        {
            node_msgs[i].clear();
            vec_mask[i]  = nodes_[i]->GetQueMsg(node_msgs[i]);
            if(0 != vec_mask[i])
                remain_count ++;
        }
        if(0 == remain_count)
            break;
        for(i = 0 ; i < node_count ; i ++)
        {
            int inport_count = nodes_[i]->GetInPortCount();
            if(0 != vec_mask[i])
            {
                for(j = 0 ; j < inport_count ; j ++)
                {
                    if(0 == (vec_mask[i] & (1U << j)))
                    {
                        Message msg;
                        nodes_[i]->PutMsg(j, msg);
                    }
                }
            }
        }
        graph_.wait_for_all();
    } 

    //restore loop_init_port
    for(i = 0 ; i < input_init_node_count ; i ++)
    {
        int                                 node_idx          = inputs_init_[i].first;
        std::vector<InputInitParam> const&  inputs_init_param = inputs_init_[i].second;
        int                                 inputs_init_count = (int)(inputs_init_param.size());
        for(k = 0 ; k < inputs_init_count ; k ++)
        {
            if(true == std::get<1>(inputs_init_msg[i][k]))
            {
                Message msg       = std::get<2>(inputs_init_msg[i][k]);
                int     input_idx = inputs_init_param[k].input_idx_;
                nodes_[node_idx]->PutMsg(input_idx, msg);
            }
        }
    }
    graph_.wait_for_all();

    for(i = 0 ; i < node_count ; i ++)
        nodes_[i]->SetChecking(false);

    return 0;
}

}//namespace vision_graph