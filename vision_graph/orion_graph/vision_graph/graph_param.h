#ifndef  GRAPH_PARAM_H__
#define  GRAPH_PARAM_H__

#include <string>
#include <vector>

namespace vision_graph
{

class LinkParam
{
public:
    LinkParam() noexcept : node_idx_(0), port_idx_(0) {;};
    LinkParam(int node_idx, int port_idx) noexcept : node_idx_(node_idx), port_idx_(port_idx) {;};
    LinkParam(LinkParam const& other) noexcept : node_idx_(other.node_idx_), port_idx_(other.port_idx_) {;};
    LinkParam& operator=(LinkParam const& other)
    {
        if(&other == this)
            return *this;
        
        node_idx_  = other.node_idx_;
        port_idx_  = other.port_idx_;
        return *this;
    };
    ~LinkParam() noexcept {;};
    int        node_idx_;
    int        port_idx_;
};

class InPortParam
{
public:
    InPortParam() noexcept:tensor_names_(0), links_(0) {;};
    InPortParam(InPortParam const& other) noexcept:tensor_names_(other.tensor_names_), 
                                                             links_(other.links_){;};
    InPortParam& operator=(InPortParam const& other)
    {
        if(&other == this)
            return *this;
        
        tensor_names_  = other.tensor_names_;
        links_         = other.links_;
        return *this;
    };

    ~InPortParam() noexcept {;};
public:
    std::vector<std::string>   tensor_names_;
    std::vector<LinkParam>     links_;
};

class OutPortParam
{
public:
    OutPortParam() noexcept:tensor_name_(""), links_(0) {;};
    OutPortParam(OutPortParam const& other) noexcept:tensor_name_(other.tensor_name_), 
                                                     links_(other.links_){;};
    OutPortParam& operator=(OutPortParam const& other)
    {
        if(&other == this)
            return *this;
        
        tensor_name_  = other.tensor_name_;
        links_        = other.links_;
        return *this;
    };

    ~OutPortParam() noexcept {;};
public:
    std::string                tensor_name_;
    std::vector<LinkParam>     links_;
};

class InputInitParam
{
public:
    InputInitParam() noexcept:input_idx_(0), init_tensor_name_("") {;};
    InputInitParam(InputInitParam const& other) noexcept:input_idx_(other.input_idx_), 
                                                         init_tensor_name_(other.init_tensor_name_){;};
    InputInitParam& operator=(InputInitParam const& other)
    {
        if(&other == this)
            return *this;
        
        input_idx_         = other.input_idx_;
        init_tensor_name_  = other.init_tensor_name_;
        return *this;
    };

    ~InputInitParam() noexcept {;};
public:
    int                  input_idx_;
    std::string          init_tensor_name_;
};

class SolverParam
{
public:
    SolverParam() noexcept : solver_name_(""), class_("") {;};
    SolverParam(SolverParam const& other) noexcept : solver_name_(other.solver_name_),
                                                     class_(other.class_),
                                                     friends_(other.friends_){;};
    SolverParam& operator=(SolverParam const& other)
    {
        if(&other == this)
            return *this;
        
        solver_name_     = other.solver_name_;
        class_           = other.class_;
        friends_         = other.friends_;
        return *this;
    };
    ~SolverParam() noexcept {;};

    std::string                solver_name_;
    std::string                class_;
    std::vector<std::string>   friends_;
};

class NodeParam
{
public:
    NodeParam() noexcept:node_name_(""), inputs_(0), outputs_(0), enable_ports_(0), inputs_init_(0) {;};
    NodeParam(NodeParam const& other) noexcept:node_name_(other.node_name_), 
                                               inputs_(other.inputs_), 
                                               outputs_(other.outputs_), 
                                               enable_ports_(other.enable_ports_),
                                               inputs_init_(other.inputs_init_),
                                               solver_(other.solver_) {;};
    NodeParam& operator=(NodeParam const& other)
    {
        if(&other == this)
            return *this;
        
        node_name_     = other.node_name_;
        inputs_        = other.inputs_;
        outputs_       = other.outputs_;
        enable_ports_  = other.enable_ports_;
        inputs_init_   = other.inputs_init_;
        solver_        = other.solver_;
        return *this;
    };

    ~NodeParam() noexcept {;};
public:
    std::string                    node_name_;
    std::vector<InPortParam>       inputs_;
    std::vector<OutPortParam>      outputs_;
    std::vector<InPortParam>       enable_ports_;
    std::vector<InputInitParam>    inputs_init_;
    SolverParam                    solver_;
};

class SubGraphParam
{
public:
    SubGraphParam() noexcept:sub_graph_name_(""), print_profile_(false), nodes_(0) {;};
    SubGraphParam(SubGraphParam const& other) noexcept:sub_graph_name_(other.sub_graph_name_), 
                                                       print_profile_(other.print_profile_),
                                                       nodes_(other.nodes_){;};
    SubGraphParam& operator=(SubGraphParam const& other)
    {
        if(&other == this)
            return *this;
        
        sub_graph_name_  = other.sub_graph_name_;
        print_profile_   = other.print_profile_;
        nodes_           = other.nodes_;
        return *this;
    };

    ~SubGraphParam() noexcept {;};
public:
    std::string                       sub_graph_name_;
    bool                              print_profile_;
    std::vector<NodeParam>            nodes_;
};

}//namespace vision_graph

#endif