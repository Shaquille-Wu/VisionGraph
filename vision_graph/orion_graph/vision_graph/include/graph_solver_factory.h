/*
 * Copyright (C) OrionStart Technology(Beijing) Co., Ltd. - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 */

/**
 * @file graph_solver_factory.h
 * @brief This header file defines solver factory.
 * @author WuXiao(wuxiao@ainirobot.com)
 * @date 2020-07-06
 */

#ifndef GRAPH_SOLVER_FACTORY_H_
#define GRAPH_SOLVER_FACTORY_H_

#include <string>
#include <unordered_map>
#include <typeinfo>
#include <functional>
#include <cxxabi.h>
#include "graph_solver.h"

namespace vision_graph
{
class SolverFactory
{
public:
    static SolverFactory* Instance() noexcept
    {
        if (nullptr == solver_factory_instance_)
        {
            solver_factory_instance_ = new SolverFactory();
        }
        return solver_factory_instance_;
    };

    virtual ~SolverFactory() noexcept
    {
    };

    bool Regist(const std::string& class_name, std::function<Solver*(nlohmann::json const& param)> create_func) noexcept
    {
        if (nullptr == create_func || std::string("") == class_name)
        {
            return false;
        }
        bool reg = map_create_func_.insert(std::make_pair(class_name, create_func)).second;

        return reg;
    }

    Solver* Create(const std::string& class_name, nlohmann::json const& param) noexcept
    {
        auto iter = map_create_func_.find(class_name);
        if (iter == map_create_func_.end())
        {
            return nullptr;
        }
        else
        {
            return (iter->second(param));
        }
    }

    bool    Check(const std::string& class_name) const noexcept
    {
        auto iter = map_create_func_.find(class_name);
        if (iter == map_create_func_.end())
            return false;
        return true;
    }

private:
    SolverFactory() noexcept {;} ;
    SolverFactory(const SolverFactory& other) = delete;
    SolverFactory(SolverFactory&& other) = delete;
    SolverFactory& operator=(const SolverFactory& other) = delete;
    SolverFactory& operator=(SolverFactory&& other) = delete;
    static SolverFactory*                                                                  solver_factory_instance_;   
    std::unordered_map<std::string, std::function<Solver*(nlohmann::json const& param)> >  map_create_func_;
};

template<typename T>
class SolverCreator
{
public:
    class Register
    {
    public:
        Register() noexcept
        {
            char* demangle_name = nullptr;
            std::string type_name;
            demangle_name = abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr);
            if (nullptr != demangle_name)
            {
                type_name = demangle_name;
                free(demangle_name);
            }
            solver_class_name_ = type_name;
            SolverFactory::Instance()->Regist(type_name, CreateSolver);
        };

        inline void do_nothing()const noexcept
        { 
        };
     };
     SolverCreator() noexcept
     {
          solver_register_.do_nothing();
     }
     virtual ~SolverCreator() noexcept
     {
          solver_register_.do_nothing();
     };

     static T* CreateSolver(nlohmann::json const& param) noexcept
     {
          return new T(param);
     }
     static Register     solver_register_;
     static std::string  solver_class_name_;

private:
    SolverCreator(const SolverCreator& other) = delete;
    SolverCreator(SolverCreator&& other) = delete;
    SolverCreator& operator=(const SolverCreator& other) = delete;
    SolverCreator& operator=(SolverCreator&& other) = delete;         
};
template<typename T>
typename vision_graph::SolverCreator<T>::Register vision_graph::SolverCreator<T>::solver_register_;
template<typename T>
typename std::string vision_graph::SolverCreator<T>::solver_class_name_;
} //namespace vision_graph

#endif