/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file vision_graph.cpp
 * @brief This source file implement graph's detail.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */
#include "vision_graph.hpp"
#include <detector.h>
#include <keypoints_attributes.h>
#include <reidfeature.h>

vision_graph::SolverFactory* vision_graph::SolverFactory::solver_factory_instance_ = nullptr;

namespace vision_graph
{
Graph::Graph() noexcept : graph_impl_(nullptr)
{
}

Graph::~Graph() noexcept
{
    if(nullptr != graph_impl_)
        graph_impl_.reset(nullptr);
}

int Graph::Build(std::string const& graph_file/*, SolverFactory* solver_factory*/) noexcept
{
    SolverFactory*   solver_factory = vision_graph::SolverFactory::Instance();
    if(nullptr == solver_factory)   return -1;

    nlohmann::json  graph_json;
    std::ifstream   input_stream(graph_file);
    if(false == input_stream.is_open())
        return kErrCodeJsonInvalid;
    input_stream >> graph_json;

    int             i              = 0;
    bool            res            = 0;
    if(false == graph_json.contains(kJsonSubGraphSet))
        return -1104;

    //1. we should parse all of nodes in all of sub_graphs
    std::vector<nlohmann::json>                sub_graphs_json   = graph_json.at(kJsonSubGraphSet);
    int                                        sub_graph_count   = (int)(sub_graphs_json.size());
    std::vector<SubGraphParam>                 vec_sub_graph_params(sub_graph_count);
    std::vector<std::vector<nlohmann::json> >  vec_sub_graph_solver_json(sub_graph_count);
    for(i = 0 ; i < sub_graph_count ; i ++)
    {
        int sub_graph_res =  SubGraph::ParseSubGraph(sub_graphs_json[i], 
                                                     solver_factory, 
                                                     vec_sub_graph_params[i],
                                                     vec_sub_graph_solver_json[i]);
        if(0 != sub_graph_res)
        {
            LOG(ERROR) << "sub_graph(" << i << "), parse error " << sub_graph_res;
            ABORT();
        }
    }

    res = SubGraph::CheckSubGraphsNodes(vec_sub_graph_params);
    if(false == res)
    {
        LOG(ERROR) << "sessions check error";
        ABORT();
    }

    //2. we create solvers as global object
    graph_impl_ = std::unique_ptr<Implementation>(new Implementation);
    res = graph_impl_->CreateSolvers(vec_sub_graph_params, vec_sub_graph_solver_json, solver_factory);
    if(false == res)
    {
        LOG(ERROR) << "CreateSolvers failed";
        ABORT();
        goto BUILD_GRAPH_LEAVE;
    } 

    //3. we create tensor_outs as global object
    res = graph_impl_->CreateTensorOuts(vec_sub_graph_params);
    if(false == res)
    {
        LOG(ERROR) << "CreateTensorOuts failed";
        ABORT();
        goto BUILD_GRAPH_LEAVE;
    } 

    //4. we create graph, including tensors(tensor_out), nodes, and so on ...
    res = graph_impl_->BuildGraph(vec_sub_graph_params);
    if(false == res)
    {
        LOG(ERROR) << "BuildGraph failed";
        ABORT();
    } 

BUILD_GRAPH_LEAVE:
    if(false == res)
    {
        graph_impl_->DestroyGraph();
        graph_impl_.reset(nullptr);
        graph_impl_ = nullptr;
    }
    return (true == res ? kErrCodeOK : kErrCodeParamInvalid) ;
}

int Graph::Destroy() noexcept
{
    if(nullptr != graph_impl_)
    {
        graph_impl_->DestroyGraph();
        graph_impl_.reset(nullptr);
    }
    graph_impl_ = nullptr;

    return 0;
}

std::map<std::string, std::pair<Tensor*, Solver*> > Graph::GetTensors(int sub_graph_idx) noexcept
{
    std::map<std::string, std::pair<Tensor*, Solver*> > empty_map;
    if(nullptr == graph_impl_)   return empty_map ;

    return graph_impl_->GetTensors(sub_graph_idx);
}

std::vector<Solver*> Graph::GetSolvers(int sub_graph_idx) noexcept
{
    std::vector<Solver*>   solver(0);
    if(nullptr == graph_impl_)   return solver ;

    return graph_impl_->GetSolvers(sub_graph_idx);
}

int Graph::Start(int sub_graph_idx) noexcept
{
    if(nullptr == graph_impl_)   return kErrCodeGraphInvalid ;

    return graph_impl_->Start(sub_graph_idx);
}

void Graph::Wait(int sub_graph_idx) noexcept
{
    if(nullptr == graph_impl_)   return ;

    graph_impl_->Wait(sub_graph_idx);
}

}; //namespace vision_graph
