#include "tbb/flow_graph.h"
#include <iostream>

using namespace std;
using namespace tbb::flow;

class TestBase
{
public:
    TestBase(){;};
    virtual ~TestBase(){;};

    virtual TestBase operator+(TestBase const& other)
    {
        std::cout << "TestBase.+" << std::endl;

        return TestBase();
    }

    virtual TestBase& operator+=(TestBase const& other)
    {
        std::cout << "TestBase.+=" << std::endl;

        return *this;
    }
};

class TestA: public TestBase
{
public:
    TestA(){;};
    virtual ~TestA(){;};

    virtual TestA operator+(TestA const& other)
    {
        std::cout << "TestA.+" << std::endl;

        return TestA();
    }

    virtual TestA& operator+=(TestA const& other)
    {
        std::cout << "TestA.+=" << std::endl;

        return *this;
    }
};

class TestB : public TestBase
{
public:
    TestB(){;};
    virtual ~TestB(){;};

    virtual TestB operator+(TestB const& other)
    {
        std::cout << "TestB.+" << std::endl;
        return TestB();
    }

    virtual TestB& operator+=(TestB const& other)
    {
        std::cout << "TestB.+=" << std::endl;
        return *this;
    }
};

int main() 
{
    graph g;
    continue_node< continue_msg> hello( g,
      []( const continue_msg &) {
            cout << "Hello";
        }
    );
    continue_node< continue_msg> world( g,
      []( const continue_msg &) {
            cout << " World\n";
        }
    );
    make_edge(hello, world);
    hello.try_put(continue_msg());
    g.wait_for_all();


    TestBase*  pBase = new TestBase;
    TestBase*  pA    = new TestA;
    TestBase*  pB    = new TestB;

    TestBase&  A     = *pA ;
    TestBase&  B     = *pB ;

    TestBase   C     = A + B ;

    A += B;


    return 0;
}