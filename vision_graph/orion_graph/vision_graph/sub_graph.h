/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file sub_graph.h
 * @brief This header file defines sub graph.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-24
 */

#ifndef SUB_GRAPH_H_
#define SUB_GRAPH_H_

#include "graph_param.h"
#include "graph_func_node.h"
#include "include/graph_solver_factory.h"
#include "include/graph_error.h"

namespace vision_graph
{

static const std::string                 kSolverDataType      = "vision_graph::SolverData";

static const std::map<std::string, int>  kInputSolverMap  = {
    { "vision_graph::SolverData",     0 }
};

static const std::map<std::string, int>  kZeroOutputSolverMap = {
};

static const std::map<std::string, int>  kBranchSolverMap     = {
    { "vision_graph::SolverBranch",   0 }
};

static const char    kJsonSubGraphSet[]                        = "sub_graphs";

class SubGraph
{  
public:
    SubGraph() noexcept : input_node_idx_(0), inputs_init_(0), print_profile_(false), first_start_(true) {;};
    virtual ~SubGraph() {;};

    /**
     * @brief parse sub_graph param from json
     * 
     * @param sub_graph_json the json node(input)
     * 
     * @param sub_graph_param the result of parse(output)
     * 
     * @return the "parse" status, >= 0 is ok, other is failed
     * 
     */
    static int                                     ParseSubGraph(nlohmann::json const&         sub_graph_json, 
                                                                 SolverFactory const*          solver_factory,
                                                                 SubGraphParam&                sub_graph_param,
                                                                 std::vector<nlohmann::json>&  solvers_json) noexcept;

    /**
     * @brief parse check list of sub_graph param
     * 
     * @param sub_graphs the list of sub_graph param
     * 
     * @return the "check" status, true or false
     * 
     */
    static bool                                    CheckSubGraphsNodes(std::vector<SubGraphParam> const& sub_graphs) noexcept;

    /**
     * @brief build sub graph with sub_graph and solver_jsons
     * 
     * @param sub_graph the sub_graph's parameter, which can be load from json file
     * 
     * @param solvers_map solvers map, name to pointer
     * 
     * @return true if it success, false if it failed
     * 
     */
    bool               Build(SubGraphParam const&                                  sub_graph, 
                             std::map<std::string, Solver*>&                       name_2_solvers_map,
                             std::map<std::string, std::pair<Tensor*, Solver*> >&  tensor_outs_map) noexcept ;

    /**
     * @brief free related resource
     * 
     * @return 0 if it success, false if it failed
     * 
     */
    int                Destroy() noexcept;


    int                Start() noexcept;


    void               Wait() noexcept;

    std::string const& GetSubGraphName() const noexcept { return sub_graph_name_; };

    std::vector<std::pair<std::string, std::vector<Tensor*> > >&    GetTensorsVec() noexcept
    {
        return nodes_tensor_out_vec_;
    }

    std::map<std::string, std::pair<Tensor*, Solver*> >&            GetTensors() noexcept
    {
        return tensor_out_map_;
    }

    std::vector<Solver*>                                            GetSolvers() noexcept
    {
        return solvers_;
    }

private:
    static std::vector<InPortParam>                parse_node_inputs(std::vector<nlohmann::json> const& inputs_json) noexcept;

    static std::vector<OutPortParam>               parse_node_outputs(std::vector<nlohmann::json> const& outputs_json) noexcept;

    static std::vector<InputInitParam>             parse_inputs_init(std::vector<nlohmann::json> const& inputs_init_json) noexcept;

    static SolverParam                             parse_node_solver(nlohmann::json const& solver_json) noexcept;

    static bool                                    check_nodes_input_output(std::vector<NodeParam> const&      graph_nodes,
                                                                            std::map<std::string, LinkParam>&  out_tensors_map) noexcept;

    static bool                                    check_node_input(std::vector<NodeParam> const&              graph_nodes,
                                                                    std::map<std::string, LinkParam> const&    out_tensors_map,
                                                                    bool                                       is_enable_ports) noexcept;
    
    static bool                                    inference_nodes_link(std::vector<NodeParam>&                  graph_nodes, 
                                                                        std::map<std::string, LinkParam> const&  out_tensors_map) noexcept;

    static bool                                    check_sub_graph(std::vector<NodeParam> const&             graph_nodes, 
                                                                   std::map<std::string, int> const&         nodes_name_idx_map, 
                                                                   SolverFactory const*                      solver_factory) noexcept;

    bool                                           contain_reference_input(std::vector<Tensor*> const& inputs, std::vector<std::string> const& input_names) noexcept;

    GraphFuncNode*                                 create_func_node(std::string const&           node_name,
                                                                    int                          total_in_ports, 
                                                                    int                          enable_ports,
                                                                    int                          out_ports, 
                                                                    Solver*                      solver,
                                                                    std::vector<Tensor*> const&  out_tensors) noexcept;

    std::vector<int>                                           find_input_nodes(std::vector<NodeParam> const&  graph_nodes) noexcept;

    std::vector<std::pair<int, std::vector<InputInitParam> > > find_input_init_nodes(std::vector<NodeParam> const&  graph_nodes) noexcept;

    int                                            clear_msg_que() noexcept;

protected:
    /**
     * @brief tbb's graph
     *
     */
    tbb::flow::graph                                                    graph_;

    /**
     * @brief sub_graph's name
     *
     */
    std::string                                                         sub_graph_name_;

    /**
     * @brief nodes list
     *
     */
    std::vector<std::unique_ptr<GraphFuncNode> >                        nodes_;

    /**
     * @brief solver list
     *
     */
    std::vector<Solver*>                                                solvers_;

    /**
     * @brief tensor vector, <node_name, tensor_outs>
     *
     */
    std::vector<std::pair<std::string, std::vector<Tensor*> > >         nodes_tensor_out_vec_;

    /**
     * @brief tensor map, <node_name, tensor_out>
     *
     */
    std::map<std::string, std::vector<Tensor*> >                        nodes_tensor_out_map_;


    /**
     * @brief tensor map, <node_name, <tensor_out, solver> >
     *
     */
    std::map<std::string, std::pair<Tensor*, Solver*> >                 tensor_out_map_;

    /**
     * @brief input node idx
     *
     */
    std::vector<int>                                                    input_node_idx_;

    /**
     * @brief input_init
     *
     */
    std::vector<std::pair<int, std::vector<InputInitParam> > >          inputs_init_;

    /**
     * @brief flag for whether print profile information or not
     *
     */
    bool                                                                print_profile_;

    /**
     * @brief flag for first start
     *
     */
    bool                                                                first_start_;

    /**
     * @brief nodes' parameter from json
     *
     */
    std::vector<NodeParam>                                              nodes_param_;
};

}//namespace vision_graph

#endif