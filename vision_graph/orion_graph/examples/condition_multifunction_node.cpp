#include "tbb/flow_graph.h"
#include "tbb/concurrent_vector.h"
#include <iostream>
#include <thread>


typedef tbb::flow::multifunction_node<int, std::tuple<int, int> > mf_node;

struct mfunc_body 
{
public:
    mfunc_body(const char* name)
    {
        std::cout << "mfunc_body ctor " << name << std::endl;
    };

    virtual ~mfunc_body()
    {

    };

    mfunc_body(const mfunc_body &other )
    { ; };

    void operator=( const mfunc_body& )
    {
        ;
    };

    void operator()(const int& in, mf_node::output_ports_type &op) 
    {
        bool send_to_one = in > 0;
        if(send_to_one) 
            std::get<0>(op).try_put(in);
        else            
            std::get<1>(op).try_put(in);
    }
};

struct otherbody {
    otherbody()
    {

    };
    int operator()(const int& in) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return in;
    }
};

int main() 
{
    tbb::flow::graph g;
    //tbb::flow::source_node< int > start0(g);
    //tbb::flow::source_node< int > start1(g);
    //tbb::flow::function_node<int, int> f1(g, tbb::flow::unlimited, otherbody());
    //tbb::flow::function_node<int, int> f2(g, tbb::flow::unlimited, otherbody());

    tbb::flow::queue_node<int> even_queue(g);
    tbb::flow::queue_node<int> odd_queue(g);
    mf_node cn(g, tbb::flow::unlimited, mfunc_body("abc"));

    //mf_node cn(g, tbb::flow::unlimited, mfunc_body("abc"));

    tbb::flow::output_port<0>(cn).register_successor(even_queue);
    tbb::flow::make_edge(tbb::flow::output_port<1>(cn), even_queue);

    //start0.try_put(0);
    //start1.try_put(2);
    cn.try_put(1);
    cn.try_put(2);
    g.wait_for_all();
    int o = 0, e = 0;
    bool res0 = odd_queue.try_get(o);
    bool res1 = even_queue.try_get(e);
    std::cout << "exit" << std::endl;
}

/*
using namespace tbb::flow;

tbb::spin_mutex io_lock;
typedef broadcast_node<int> bnode_element_t;
typedef tbb::concurrent_vector<bnode_element_t *> output_port_vector_t;
struct multioutput_function_body {
    output_port_vector_t &my_ports;
    public:
    multioutput_function_body(output_port_vector_t &_ports) : my_ports(_ports) {}
    multioutput_function_body(const multioutput_function_body &other) : my_ports(other.my_ports) { }
    continue_msg operator()(const int in) {
        int current_size = my_ports.size();
        if(in >= current_size) {
            // error condition?  grow concurrent_vector?
            tbb::spin_mutex::scoped_lock gl(io_lock);
            std::cout << "Received input out of range(" << in << ")" << std::endl;
        }
        else {
            // do computation
            my_ports[in]->try_put(in*2);
        }
        return continue_msg();
    }
};

struct output_function_body {
    int my_prefix;
    output_function_body(int i) : my_prefix(i) { }
    int operator()(const int i) {
        tbb::spin_mutex::scoped_lock gl(io_lock);
        std::cout << " output node "<< my_prefix << " received " << i << std::endl;
        return i;
    }
};

int main() {
    graph g;
    output_port_vector_t output_ports;
    function_node<int> my_node(g, unlimited, multioutput_function_body(output_ports) );
    // create broadcast_nodes
    for( int i = 0; i < 20; ++i) {
        bnode_element_t *bp = new bnode_element_t(g);
        output_ports.push_back(bp);
    }

    // attach the output nodes to the broadcast_nodes
    for(int i = 0; i < 20; ++i) {
        function_node<int,int> *fp = new function_node<int,int>(g, unlimited, output_function_body(i));
        make_edge(*(output_ports[i]),*fp);
    }

    for( int i = 0; i < 21; ++i) {
        my_node.try_put(i);
    }
    g.wait_for_all();
    return 0;
}
*/