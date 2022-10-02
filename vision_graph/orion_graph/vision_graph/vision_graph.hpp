/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file vision_graph.hpp
 * @brief This head file for implement graph's detail.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-24
 */
#ifndef VISION_GRAPH_HPP_
#define VISION_GRAPH_HPP_

#include "include/vision_graph.h"
#include "include/graph_error.h"
#include <logging.h>
#include "sub_graph.h"
#include <fstream>
#include <iostream>

namespace vision_graph
{
/**
 * @brief The implementation for graph.
 *
 */
class Graph::Implementation
{
public:
    /**
     * @brief ctor
     */
    Implementation() noexcept {;};

    /**
     * @brief dtor
     */
    ~Implementation() noexcept {;};

    /**
     * @brief create solvers as global object
     * 
     * @param sub_graphs the sub_graphs's parameter, which can be load from json file
     * 
     * @param solver_jsons solvers' parameter in json
     * 
     * @param solver_factory solvers' factory
     * 
     * @return true if it success, false if it failed
     * 
     */
    bool                CreateSolvers(std::vector<SubGraphParam> const&                  sub_graphs, 
                                      std::vector<std::vector<nlohmann::json> > const&   solver_jsons,
                                      SolverFactory*                                     solver_factory) noexcept
    {
        name_2_solvers_map_.clear();
        int  i = 0, j = 0, k = 0;
        int  sub_graph_count = (int)(sub_graphs.size());

        std::map<std::string, bool>   has_friends;
        for(i = 0 ; i < sub_graph_count ; i ++)
        {
            const SubGraphParam&                sub_graph    = sub_graphs[i];
            const std::vector<nlohmann::json>&  solvers_json = solver_jsons[i];
            int   node_count                                 = (int)(sub_graph.nodes_.size());
            for(j = 0 ; j < node_count ; j ++)
            {
                const nlohmann::json&  cur_json          = solvers_json[j];
                const SolverParam&     cur_solver_param  = sub_graph.nodes_[j].solver_;
                const std::string&     solver_name       = cur_solver_param.solver_name_;
                const std::string&     solver_class      = cur_solver_param.class_;
                if(name_2_solvers_map_.end() == name_2_solvers_map_.find(solver_name))
                {
                    Solver*                solver            = solver_factory->Create(solver_class, cur_json);
                    solver->SetSolverName(solver_name);
                    name_2_solvers_map_[solver_name] = solver;
                    has_friends[solver_name]         = false;
                }
            }
        }

        for(i = 0 ; i < sub_graph_count ; i ++)
        {
            const SubGraphParam&   sub_graph  = sub_graphs[i];
            int                    node_count = (int)(sub_graph.nodes_.size());
            for(j = 0 ; j < node_count ; j ++)
            {
                const SolverParam&  cur_solver_param  = sub_graph.nodes_[j].solver_;
                const std::string&  cur_solver_name   = cur_solver_param.solver_name_;
                int                 friends_count     = (int)(cur_solver_param.friends_.size());
                if(friends_count > 0)
                {
                    std::map<std::string, bool>::iterator  iter = has_friends.find(cur_solver_name);
                    if(has_friends.end() != iter && false == iter->second)
                    {
                        Solver*                cur_solver = name_2_solvers_map_[cur_solver_name];
                        std::vector<Solver*>   friends(friends_count);
                        for(k = 0 ; k < friends_count ; k ++)
                        {
                            friends[k] = nullptr;
                            const std::string&                              cur_friend_name = cur_solver_param.friends_[k];
                            std::map<std::string, Solver*>::const_iterator  iter_solver     = name_2_solvers_map_.find(cur_friend_name);
                            if(name_2_solvers_map_.end() != iter_solver)
                                friends[k] = iter_solver->second;
                        }
                        cur_solver->SetFriendSolvers(friends);
                        iter->second = true;
                    }
                }
            }
        }

        return true;
    };

    /**
     * @brief create solvers as global object
     * 
     * @param sub_graphs the sub_graph's parameter, which can be load from json file
     * 
     * @return true if it success, false if it failed
     * 
     */
    bool                CreateTensorOuts(std::vector<SubGraphParam> const&  sub_graphs) noexcept
    {
        int    k               = 0;
        int    sub_graph_count = (int)(sub_graphs.size());
        name_2_tensor_outs_map_.clear();
        for(k = 0; k < sub_graph_count ; k ++)
        {
            const std::vector<NodeParam>&  graph_nodes = sub_graphs[k].nodes_;
            int                            node_count  = (int)(graph_nodes.size());
            int  i    = 0;
            int  j    = 0;
            for(i = 0 ; i < node_count ; i ++)
            {
                const SolverParam&  solver_param = graph_nodes[i].solver_;
                const std::string&  solver_name  = solver_param.solver_name_;
                std::map<std::string, Solver*>::iterator  iter = name_2_solvers_map_.find(solver_name);
                if(iter == name_2_solvers_map_.end())
                    return false;
                Solver*    solver            = iter->second;
                int        out_tensors_count = (int)(graph_nodes[i].outputs_.size());
                for(j = 0 ; j < out_tensors_count ; j ++)
                {
                    std::string const&                                            tensor_out_name = graph_nodes[i].outputs_[j].tensor_name_;
                    std::map<std::string, std::pair<Tensor*, Solver*> >::iterator iter_tensor     = name_2_tensor_outs_map_.find(tensor_out_name);
                    if(iter_tensor != name_2_tensor_outs_map_.end())
                        continue;
                    Tensor*  tensor_data = solver->CreateOutTensor(j);
                    assert(nullptr != tensor_data);
                    if(nullptr == tensor_data)
                        return false;
                    name_2_tensor_outs_map_[tensor_out_name] = std::make_pair(tensor_data, solver);
                }
            }
        }

        return true;
    };

    /**
     * @brief build graph with sub_graphs' parameter
     * 
     * @param sub_graphs the sub_graph's parameter, which can be load from json file
     * 
     * @return true if it success, false if it failed
     * 
     */
    bool                BuildGraph(std::vector<SubGraphParam> const&  sub_graphs) noexcept
    {
        int  i               = 0 ;
        int  sub_graph_count = (int)(sub_graphs.size());
        bool res             = true;
        sub_graphs_.resize(sub_graph_count);
        for(i = 0 ; i < sub_graph_count ; i ++)
        {
            sub_graphs_[i] = std::unique_ptr<SubGraph>(new SubGraph);
            res            = sub_graphs_[i]->Build(sub_graphs[i], name_2_solvers_map_, name_2_tensor_outs_map_);
            if(false == res)
                return false;
        }
        sub_graph_params_ = sub_graphs;
        return true;
    };

    int                 DestroyGraph() noexcept
    {
        int  i               = 0 ;
        int  sub_graph_count = (int)(sub_graphs_.size());

        for(i = 0 ; i < sub_graph_count ; i ++)
        {
            sub_graphs_[i]->Destroy();
            sub_graphs_[i].reset(nullptr);
        }
        sub_graphs_.clear();

        for(auto& tensor_out:name_2_tensor_outs_map_)
        {
            if(nullptr != (tensor_out.second).first)
                delete (tensor_out.second).first;
            (tensor_out.second).first = nullptr;
        }
        name_2_tensor_outs_map_.clear();

        for(auto& solver:name_2_solvers_map_)
        {
            if(nullptr != solver.second)
                delete solver.second;
            solver.second = nullptr;
        }
        name_2_solvers_map_.clear();

        return 0;
    };

    std::vector<std::string>                                GetSubGraphs() const noexcept
    {
        std::vector<std::string>   vec_sub_graph_name(sub_graphs_.size());
        size_t                     i = 0;                        
        for(i = 0 ; i < sub_graphs_.size() ; i ++)
        {
            vec_sub_graph_name[i] = sub_graphs_[i]->GetSubGraphName();
        }

        return vec_sub_graph_name;
    }

    std::vector<Solver*>                                    GetSolvers(int sub_graph_idx) noexcept
    {
        return sub_graphs_[sub_graph_idx]->GetSolvers();
    };

    std::map<std::string, std::pair<Tensor*, Solver*> >     GetTensors(int sub_graph_idx) noexcept
    {
        return sub_graphs_[sub_graph_idx]->GetTensors();
    };

    int                                                     Start(int sub_graph_idx) noexcept
    {
        return sub_graphs_[sub_graph_idx]->Start();
    };

    int                                                     Wait(int sub_graph_idx) noexcept
    {
        sub_graphs_[sub_graph_idx]->Wait();
        return 0;
    };

    std::vector<SubGraphParam> const&                       GetGraphParam() const noexcept
    {
        return sub_graph_params_;
    }

private:
    std::vector<std::unique_ptr<SubGraph> >                sub_graphs_;

    /**
     * @brief solvers map, name to pointer
     *
     */
    std::map<std::string, Solver*>                         name_2_solvers_map_;

    /**
     * @brief tensor_outs map, name to pair<Tensor*, Solver*>
     *
     */
    std::map<std::string, std::pair<Tensor*, Solver*> >    name_2_tensor_outs_map_;

    /**
     * @brief the parameter read from json file
     *
     */
    std::vector<SubGraphParam>                             sub_graph_params_;
};  // class Implementation

}; //namespace vision_graph

#endif
