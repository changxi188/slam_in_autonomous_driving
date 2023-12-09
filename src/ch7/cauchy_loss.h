#pragma once

#include "common/eigen_types.h"

namespace sad
{
class CauchyLoss
{
public:
    explicit CauchyLoss(double delta) : delta_(delta)
    {
    }

    void Compute(double err2, Eigen::Vector3d& rho) const
    {
        double dsqr     = delta_ * delta_;                  // c^2
        double dsqrReci = 1. / dsqr;                        // 1/c^2
        double aux      = dsqrReci * err2 + 1.0;            // 1 + e^2/c^2
        rho[0]          = dsqr * log(aux);                  // c^2 * log( 1 + e^2/c^2 )
        rho[1]          = 1. / aux;                         // rho'
        rho[2]          = -dsqrReci * std::pow(rho[1], 2);  // rho''
    }

    void RobustInfo(const VecXd& residual, const Eigen::MatrixXd information, double& drho, Eigen::MatrixXd& info) const
    {
        Eigen::MatrixXd sqrt_information_ = Eigen::LLT<Eigen::MatrixXd>(information).matrixL().transpose();
        double          e2                = residual.transpose() * information * residual;
        Eigen::Vector3d rho;
        Compute(e2, rho);
        Eigen::VectorXd weight_err = sqrt_information_ * residual;

        Eigen::MatrixXd robust_info(information.rows(), information.cols());
        robust_info.setIdentity();
        robust_info *= rho[1];
        if (rho[1] + 2 * rho[2] * e2 > 0.)
        {
            robust_info += 2 * rho[2] * weight_err * weight_err.transpose();
        }

        info = robust_info * information;
        drho = rho[1];
    }

private:
    double delta_;
};
}  // namespace sad
