#ifndef _GRAPH_NORM_H_
#define _GRAPH_NORM_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
template<typename T>
void norm(std::vector<T>& inputs){
    double sum = 0.0;
    for(auto val : inputs){
        sum += (val * val);
    }

    double dist = sqrt(sum);

    std::for_each(inputs.begin(), inputs.end(), [dist](T& val){
                val /= dist; 
            });
}

#endif
