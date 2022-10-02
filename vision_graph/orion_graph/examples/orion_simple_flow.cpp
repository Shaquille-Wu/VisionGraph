#include "tbb/flow_graph.h"
#include <iostream>
#include <unistd.h>

struct body
{   
    std::string my_name;
    body( const char *name ) : my_name(name)
    {

    }

    bool operator()( bool avail ) const
    {   
        if (!avail)
            printf("%s skipped\n", my_name.c_str());
        else
        {
            if (my_name == "B")
            {   
                printf("%s fail\n", my_name.c_str());
                avail = false;  // fail task
            }
            else
            {   
                sleep(1);
                printf("%s\n", my_name.c_str());
            }
        }
        return avail;
    }
};

int main()
{
    tbb::flow::graph g;

    typedef tbb::flow::function_node<bool, bool> process_node;
    typedef std::tuple<bool,bool> bool_pair;
    tbb::flow::broadcast_node< bool > start(g);
    process_node a( g, tbb::flow::unlimited, body("A"));
    process_node b( g, tbb::flow::unlimited, body("B"));
    process_node c( g, tbb::flow::unlimited, body("C"));
    tbb::flow::join_node<bool_pair> join_c(g);
    tbb::flow::function_node<bool_pair, bool> and_c(g, tbb::flow::unlimited, [](const bool_pair& in)->bool {
        return std::get<0>(in) && std::get<1>(in);
    });
    process_node d( g, tbb::flow::unlimited, body("D"));
    process_node e( g, tbb::flow::unlimited, body("E"));

    /*
     * start -+-> A -+-> E
     *        |       \
     *        |        \
     *        |         join_c -> and_c -> C -> D
     *        |        /
     *        |       /
     *        +-> B -- 
     */
    tbb::flow::make_edge( start, a );
    tbb::flow::make_edge( start, b );
    tbb::flow::make_edge( a, tbb::flow::input_port<0>(join_c) );
    tbb::flow::make_edge( b, tbb::flow::input_port<1>(join_c) );
    tbb::flow::make_edge( join_c, and_c );
    tbb::flow::make_edge( and_c, c );
    tbb::flow::make_edge( c, d );
    tbb::flow::make_edge( a, e );

    

    for (int i = 0; i < 3; ++i )
    {
        try
        {   
            start.try_put( true );
            g.wait_for_all();
        } 
        catch (...)
        {   
            printf("Caught exception\n");
        }
    }

    return 0;
}