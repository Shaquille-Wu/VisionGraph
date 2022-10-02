/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file vision_graph.h
 * @brief This header file defines basic graph.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef VISION_GRAPH_H_
#define VISION_GRAPH_H_

#include "graph_solver.h"

namespace vision_graph
{

/**
 * @brief The definition for graph.
 *
 */
class Graph
{
public:
    /**
     * @brief ctor
     * 
     */
    Graph() noexcept;

    /**
     * @brief dtor
     * 
     */
    virtual ~Graph() noexcept;

    /**
     * @brief build graph with json file, which define the graph with various of nodes
     * 
     * create reference resources, including nodes, solvers, and tensors
     * 
     * @param graph_file the json file name
     * 
     * @return the "Build" status, >= 0 is ok, other is failed
     * 
     */
    virtual int                                             Build(std::string const& graph_file) noexcept;

    /**
     * @brief destroy graph, including nodes, solvers, and tensors
     * 
     * @return the "Destroy" status, >= 0 is ok, other is failed
     * 
     */
    virtual int                                             Destroy() noexcept;

    /**
     * @brief get all of sub_graphs' name
     * 
     * @return vector of sub_graph's name
     * 
     */
    std::vector<std::string>                                GetSubGraphs() const noexcept;

    /**
     * @brief get all of sub_graph's solvers
     * 
     * @return the solvers list, sorted by nodes' sequence
     * 
     */
    std::vector<Solver*>                                    GetSolvers(int sub_graph_idx) noexcept;

    /**
     * @brief get all of sub_graph's tensor(out), which created by graph
     * 
     * @return the tensors map
     * 
     * @note std::string  tensor_name 
     *       Tensor*      tensor_ptr 
     *       Solver*      solver_ptr, which created this tensor
     * 
     */
    std::map<std::string, std::pair<Tensor*, Solver*> >     GetTensors(int sub_graph_idx) noexcept;

    /**
     * @brief trigger the graph to work, start running
     * 
     * @param sub_graph_idx index of sub_graph
     * 
     * @return the status of "Start", >= 0 is ok, other is failed
     * 
     */
    int                                                     Start(int sub_graph_idx) noexcept;

    /**
     * @brief wait for all nodes' computation are finished
     * 
     * @param sub_graph_idx index of sub_graph
     * 
     * @return none
     * 
     */
    void                                                    Wait(int sub_graph_idx) noexcept;

protected:
    class Implementation;
    std::unique_ptr<Implementation>                         graph_impl_;

};//class Graph

}//namespace vision_graph


#endif