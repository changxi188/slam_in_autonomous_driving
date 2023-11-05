#pragma once

#include <g2o/core/base_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/optimizable_graph.h>
#include <g2o/core/dynamic_aligned_buffer.hpp>

const double delta  = 1e-9;
const double scalar = 1 / (2 * delta);

typedef Eigen::MatrixXd::MapType JacobianType;

/** @brief 使用数值雅克比对解析雅克比进行校验，判断二者是否相近，如果相差太多则会报错。
 *
 *  @param D                    Edge的残差维度
 *  @param E                    Edge的残差类型
 *  @param J                    存放解析雅克比的容器,这里一边就是一个std::vector
 *  @param edge                 进行校验的边
 *  @param analytic_jacobian    所有的解析雅克比
 *  @param edge_name            边的名字
 *  @param debug_log            是否显示debug信息
 */
template <int D, typename E, typename J>
void CheckJacobian(g2o::BaseEdge<D, E>* edge, const J& analytic_jacobian, std::string edge_name, bool debug_log = false)
{
    g2o::HyperGraph::VertexContainer    vertices           = edge->vertices();
    const int                           error_dim          = edge->Dimension;
    Eigen::VectorXd                     errorBeforeNumeric = edge->error();
    Eigen::VectorXd                     errorBak;
    g2o::dynamic_aligned_buffer<double> buffer{12};
    std::vector<Eigen::MatrixXd>        numerical_jacobians;
    numerical_jacobians.resize(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        g2o::OptimizableGraph::Vertex* v = static_cast<g2o::OptimizableGraph::Vertex*>(vertices[i]);
        assert(v->dimension() >= 0);
        numerical_jacobians.at(i).resize(D, v->dimension());
    }

    for (size_t i = 0; i < vertices.size(); ++i)
    {
        // Xi - estimate the jacobian numerically
        g2o::OptimizableGraph::Vertex* vi = static_cast<g2o::OptimizableGraph::Vertex*>(vertices[i]);

        if (vi->fixed())
        {
            continue;
        }
        else
        {
            const int vi_dim = vi->dimension();
            assert(vi_dim >= 0);

            double* add_vi = buffer.request(vi_dim);

            std::fill(add_vi, add_vi + vi_dim, 0.0);
            assert(error_dim >= 0);
            assert(numerical_jacobians[i].rows() == error_dim && numerical_jacobians[i].cols() == vi_dim &&
                   "jacobian cache dimension does not match");

            // add small step along the unit vector in each dimension
            for (int d = 0; d < vi_dim; ++d)
            {
                vi->push();
                add_vi[d] = delta;
                vi->oplus(add_vi);
                edge->computeError();
                errorBak = edge->error();
                vi->pop();
                vi->push();
                add_vi[d] = -delta;
                vi->oplus(add_vi);
                edge->computeError();
                errorBak -= edge->error();
                vi->pop();
                add_vi[d] = 0.0;

                numerical_jacobians[i].col(d) = scalar * errorBak;
            }  // end dimension
        }
    }
    edge->error() = errorBeforeNumeric;

    for (std::size_t i = 0; i < numerical_jacobians.size(); ++i)
    {
        bool is_approx = numerical_jacobians.at(i).isApprox(analytic_jacobian.at(i), 1e-3);
        if (debug_log)
        {
            std::cout << "\n \n " << edge_name << ", Vertex: " << i << " : " << std::endl;
            std::cout << "numerical_jacobians : \n" << numerical_jacobians.at(i) << std::endl;
            std::cout << "analytic_jacobians : \n" << analytic_jacobian.at(i).matrix() << std::endl;
        }

        if (!is_approx)
        {
            std::string err_string = edge_name + ", Vertex " + std::to_string(i) + " jacobian has error!!!";
            throw std::runtime_error(err_string);
        }
    }
}
