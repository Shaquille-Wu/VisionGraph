#include <iostream>
#include "string_func.h"
#include <getopt.h>
#include "../vision_graph/vision_graph.hpp"
#include <time.h>

using namespace vision_graph;

int ParseGraph(std::string const& graph_file, std::vector<SubGraphParam>& out_sub_graphs)
{
    nlohmann::json  graph_json;
    std::ifstream   input_stream(graph_file);
    if(false == input_stream.is_open())
        return -1;
    input_stream >> graph_json;

    int             i              = 0;
    bool            res            = 0;
    if(false == graph_json.contains(kJsonSubGraphSet))
        return -1;

    //1. we should parse all of nodes in all of sub_graphs
    std::vector<nlohmann::json>                sub_graphs_json   = graph_json.at(kJsonSubGraphSet);
    int                                        sub_graph_count   = (int)(sub_graphs_json.size());
    std::vector<SubGraphParam>                 vec_sub_graph_params(sub_graph_count);
    std::vector<std::vector<nlohmann::json> >  vec_sub_graph_solver_json(sub_graph_count);
    for(i = 0 ; i < sub_graph_count ; i ++)
    {
        int sub_graph_res =  SubGraph::ParseSubGraph(sub_graphs_json[i], 
                                                     nullptr, 
                                                     vec_sub_graph_params[i],
                                                     vec_sub_graph_solver_json[i]);
        if(0 != sub_graph_res)
        {
            LOG(ERROR) << "sub_graph(" << i << "), parse error";
            ABORT();
        }
    }

    res = SubGraph::CheckSubGraphsNodes(vec_sub_graph_params);
    if(false == res)
    {
        LOG(ERROR) << "sessions check error";
        ABORT();
    }

    out_sub_graphs = vec_sub_graph_params;

    return 0;
}

static std::string  GenerateDotRandomNum(int num_len)
{
    int    i = 0 ;
    char*  num_string = new char[num_len + 1];
    for(i = 0 ; i < num_len ; i ++)
        num_string[i] = ('0' + rand() % 10);
    num_string[num_len] = 0;

    std::string result = std::string(num_string);

    delete num_string;

    return result;
}

static const std::string  DOT_IMAGE_EXT                    = ".png";
static const std::string  DOT_KEYWORD_DIGRAPH              = "digraph";
static const std::string  DOT_KEYWORD_RANKDIR              = "rankdir";
static const std::string  DOT_KEYWORD_IDENT                = "    ";
static const std::string  DOT_KEYWORD_NODE                 = "node";
static const std::string  GRAPH_KEYWORD_TENSOR_IN          = "tensor_in";
static const std::string  GRAPH_KEYWORD_NODE_NAME          = "node_name";
static const std::string  GRAPH_KEYWORD_NODE_NAME_LABEL    = "node_name";
static const std::string  GRAPH_KEYWORD_SOLVER_CLASS       = "solver_class";
static const std::string  GRAPH_KEYWORD_SOLVER_NAME        = "solver_name";
static const std::string  GRAPH_KEYWORD_SOLVER_NAME_LABEL  = "solver_name";
static const std::string  GRAPH_KEYWORD_TENSOR_OUT         = "tensor_out";
static const std::string  GRAPH_RANK_DIR                   = "TB";
static const std::string  GRAPH_START_SIDE                 = "s";
static const std::string  GRAPH_END_SIDE                   = "n";

std::string  GenerateDotNode(NodeParam const& node_param)
{
    static const int  node_string_len = 16384;
    int     i                  = 0 ;
    int     node_offset        = 0 ;
    char*   node_string        = new char[node_string_len];
    int     input_total_count  = (int)(node_param.inputs_.size() + node_param.enable_ports_.size());
    int     output_count       = (int)(node_param.outputs_.size());
    std::vector<InPortParam>       input_total(input_total_count);

    for(i = 0 ; i < (int)(node_param.inputs_.size()) ; i ++)
        input_total[i] = node_param.inputs_[i];
    for(i = 0 ; i < (int)(node_param.enable_ports_.size()) ; i ++)
        input_total[i + node_param.inputs_.size()] = node_param.enable_ports_[i];

    memset(node_string, 0, node_string_len);
    node_offset = sprintf(node_string, 
                          "%s%s [ shape=%s, label=\"{ ", 
                          DOT_KEYWORD_IDENT.c_str(), 
                          node_param.node_name_.c_str(),
                          kBranchSolverMap.end() == kBranchSolverMap.find(node_param.solver_.class_) ? "record" : "Mrecord");
    if(input_total_count > 0)
    {
        if(input_total_count > 1)
            node_offset += sprintf(node_string + node_offset, " { ");

        for(i = 0 ; i < input_total_count ; i ++)
        {
            if(i < (int)(node_param.inputs_.size()))
                node_offset += sprintf(node_string + node_offset, 
                                        " <%s%d> in%d ", 
                                        GRAPH_KEYWORD_TENSOR_IN.c_str(), i, i);
            else
                node_offset += sprintf(node_string + node_offset, 
                                        " <%s%d> enable%d ", 
                                        GRAPH_KEYWORD_TENSOR_IN.c_str(), i, i - (int)(node_param.inputs_.size()));
            
            if((input_total_count > 1) && (i != (input_total_count - 1)))
                node_offset += sprintf(node_string + node_offset, "|");
        }

        if(input_total_count > 1)
            node_offset += sprintf(node_string + node_offset, " } ");
        
        node_offset += sprintf(node_string + node_offset, "|");
    }

    //node_name
    node_offset += sprintf(node_string + node_offset, " <%s> %s: %s", 
                           GRAPH_KEYWORD_NODE_NAME.c_str(), GRAPH_KEYWORD_NODE_NAME_LABEL.c_str(), node_param.node_name_.c_str());
    //solver_class
    node_offset += sprintf(node_string + node_offset, " | <%s> %s", 
                           GRAPH_KEYWORD_SOLVER_CLASS.c_str(), node_param.solver_.class_.c_str());
    //solver_name
    node_offset += sprintf(node_string + node_offset, " | <%s> %s: %s", 
                           GRAPH_KEYWORD_SOLVER_NAME.c_str(), GRAPH_KEYWORD_SOLVER_NAME_LABEL.c_str(), node_param.solver_.solver_name_.c_str());

    if(output_count > 0)
    {
        node_offset += sprintf(node_string + node_offset, " | ");
        if(output_count > 1)
        {
            node_offset += sprintf(node_string + node_offset, " { ");

            for(i = 0 ; i < output_count ; i ++)
            {
                node_offset += sprintf(node_string + node_offset, 
                                       " <%s%d> %s ", 
                                       GRAPH_KEYWORD_TENSOR_OUT.c_str(), i, node_param.outputs_[i].tensor_name_.c_str());
                if(i != (output_count - 1))
                    node_offset += sprintf(node_string + node_offset, "|");
            }

            node_offset += sprintf(node_string + node_offset, " } ");
        }
        else
        {
            node_offset += sprintf(node_string + node_offset, 
                                   " <%s0> %s ", 
                                   GRAPH_KEYWORD_TENSOR_OUT.c_str(), node_param.outputs_[0].tensor_name_.c_str());
        }
    }

    node_offset += sprintf(node_string + node_offset, " }\"];\n");
    std::string  dot_string = std::string(node_string);

    delete node_string;
    return dot_string;
}

std::string  GenerateDotNodeEdge(NodeParam const& node_param, std::vector<NodeParam> const& node_list)
{
    static const int  node_string_len = 16384;
    int     i                  = 0 ;
    int     j                  = 0 ;
    int     node_offset        = 0 ;
    char*   node_string        = new char[node_string_len];
    int     output_count       = (int)(node_param.outputs_.size());
    memset(node_string, 0, node_string_len);

    for(i = 0 ; i < output_count ; i ++)
    {
        int  link_count = (int)(node_param.outputs_[i].links_.size());
        for(j = 0 ; j < link_count ; j ++)
        {
            int   link_node_idx = node_param.outputs_[i].links_[j].node_idx_;
            int   link_port_idx = node_param.outputs_[i].links_[j].port_idx_;
            int   dst_node_input_count = (int)(node_list[link_node_idx].inputs_.size());
            node_offset += sprintf(node_string + node_offset, 
                                   "%s%s:%s%d:%s -> %s:%s%d:%s", 
                                   DOT_KEYWORD_IDENT.c_str(),
                                   node_param.node_name_.c_str(), 
                                   GRAPH_KEYWORD_TENSOR_OUT.c_str(), 
                                   i, 
                                   GRAPH_START_SIDE.c_str(),
                                   node_list[link_node_idx].node_name_.c_str(),
                                   GRAPH_KEYWORD_TENSOR_IN.c_str(),
                                   link_port_idx,
                                   GRAPH_END_SIDE.c_str());
            if(link_port_idx >= dst_node_input_count)
                node_offset += sprintf(node_string + node_offset, " [style=\"dashed\"]");
            node_offset += sprintf(node_string + node_offset, ";\n");
        }
    }

    std::string  dot_string = std::string(node_string);
    delete node_string;

    return dot_string;
}

std::string  GenerateDotInputInitEdge(NodeParam const& node_param, std::vector<NodeParam> const& node_list)
{
    static const int  node_string_len = 16384;
    int     i                  = 0 ;
    int     j                  = 0 ;
    int     k                  = 0 ;
    int     node_offset        = 0 ;
    char*   node_string        = new char[node_string_len];
    int     input_init_count   = (int)(node_param.inputs_init_.size());
    int     node_count         = (int)(node_list.size());
    memset(node_string, 0, node_string_len);
    for(i = 0 ; i < input_init_count ; i ++)
    {
        int                 input_idx            = node_param.inputs_init_[i].input_idx_;
        std::string const&  init_tensor          = node_param.inputs_init_[i].init_tensor_name_;
        bool                found                = false;
        int                 init_node_idx        = 0;
        int                 init_node_output_idx = 0;
        for(j = 0 ; j < node_count ; j ++)
        {
            int output_count = (int)(node_list[j].outputs_.size());
            for(k = 0 ; k < output_count ; k ++)
            {
                std::string const&  cur_tensor_name = node_list[j].outputs_[k].tensor_name_;
                if(init_tensor == cur_tensor_name)
                {
                    init_node_idx        = j;
                    init_node_output_idx = k;
                    found                = true;
                    break;
                }
            }
            if(true == found)
                break;
        }

        if(true == found)
        {
            node_offset += sprintf(node_string + node_offset, 
                                    "%s%s:%s%d:%s -> %s:%s%d:%s [style=\"dotted\"];\n", 
                                    DOT_KEYWORD_IDENT.c_str(),
                                    node_list[init_node_idx].node_name_.c_str(), 
                                    GRAPH_KEYWORD_TENSOR_OUT.c_str(), 
                                    init_node_output_idx, 
                                    GRAPH_START_SIDE.c_str(),
                                    node_param.node_name_.c_str(),
                                    GRAPH_KEYWORD_TENSOR_IN.c_str(),
                                    input_idx,
                                    GRAPH_END_SIDE.c_str());
        }
    }

    std::string  dot_string = std::string(node_string);
    delete node_string;

    return dot_string;
}

int DrawGraph(SubGraphParam const& graph_param, std::string const& out_img_name, std::string const& rankdir)
{
    int           i                 = 0;
    int           node_count        = (int)(graph_param.nodes_.size());
    std::string   dot_graph_string  = "";
    dot_graph_string = DOT_KEYWORD_DIGRAPH + " " + graph_param.sub_graph_name_ + "{\n";
    dot_graph_string = dot_graph_string + DOT_KEYWORD_IDENT + DOT_KEYWORD_RANKDIR + "=" + rankdir + ";\n";
    dot_graph_string = dot_graph_string + DOT_KEYWORD_IDENT + DOT_KEYWORD_NODE + "[ shape=record ];\n" ;
    for(i = 0 ; i < node_count ; i ++)
    {
        std::string  node_dot_string = GenerateDotNode(graph_param.nodes_[i]);
        dot_graph_string = dot_graph_string + node_dot_string;
    }

    for(i = 0 ; i < node_count ; i ++)
    {
        std::string  edge_dot_string = GenerateDotNodeEdge(graph_param.nodes_[i], graph_param.nodes_);
        if("" != edge_dot_string)
            dot_graph_string = dot_graph_string + edge_dot_string;
    }

    for(i = 0 ; i < node_count ; i ++)
    {
        std::string  edge_input_init_dot_string = GenerateDotInputInitEdge(graph_param.nodes_[i], graph_param.nodes_);
        if("" != edge_input_init_dot_string)
            dot_graph_string = dot_graph_string + edge_input_init_dot_string;
    }

    dot_graph_string = dot_graph_string + "}";

    char  dot_cmd[1024] = { 0 };
    int   ext_pos = out_img_name.rfind(DOT_IMAGE_EXT.c_str());
    if(ext_pos >= 1)
    {
        std::string   dot_rand_num       = GenerateDotRandomNum(10);
        std::string   dot_file_name      = out_img_name.substr(0, ext_pos);
        std::string   dot_file_path_name = dot_file_name + "_" + dot_rand_num + ".dot";
        ofstream      of(dot_file_path_name);
        if(of.is_open())
        {
            of << dot_graph_string;
            of.close();
        }
        sprintf(dot_cmd, "dot -Tpng %s_%s.dot -o %s%s", dot_file_name.c_str(), dot_rand_num.c_str(), dot_file_name.c_str(), DOT_IMAGE_EXT.c_str());
        int sys_res = system(dot_cmd);
        sprintf(dot_cmd, "rm -f %s", dot_file_path_name.c_str());
        sys_res = system(dot_cmd);
        (void)sys_res;
    }

    return 0;
}

int main(int argc, char* argv[])
{
    std::string   json_file;
    std::string   out_img_file;
    int           opt = 0;
    int           i   = 0;
    while ((opt = getopt(argc, argv, "hj:x")) != -1)
    {
        char cur_char = opt;
        switch (opt)
        {
            case 'h':
               std::cout << "\nDESCRIPTION:\n"
                         << "  -j  confige file(.json file).\n"
                         << std::endl;
                std::exit(0);
                break;
            case 'j':
                json_file    = trim_string((nullptr == optarg ? std::string("") : std::string(optarg)));
                break;                             
            default:
                std::cout  << "Invalid parameter specified." << std::endl;
                std::exit(-1);
        }
    }     

    std::cout << "json file:    " << json_file << std::endl;
    int   json_ext_pos = json_file.rfind(".json");
    if(json_ext_pos <= 0)
        out_img_file = json_file;
    else
        out_img_file = json_file.substr(0, json_ext_pos);
    std::cout << "image prefix: " << out_img_file << std::endl;

    std::vector<SubGraphParam>  sub_graphs;
    int res = ParseGraph(json_file, sub_graphs);
    if(0 != res)
    {
        std::cout  << "json file is invalid." << std::endl;
        std::exit(-1);
    }

    srand(time(0));

    int sub_graph_count = (int)(sub_graphs.size());
    for(i = 0 ; i < sub_graph_count ; i ++)
    {
        std::string   sub_graph_name = sub_graphs[i].sub_graph_name_;
        sub_graph_name = out_img_file + "_" + sub_graph_name + DOT_IMAGE_EXT;
        DrawGraph(sub_graphs[i], sub_graph_name, GRAPH_RANK_DIR);
    }

    return 0;
}