
//
// Created by xiang on 2022/7/7.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ch7/icp_3d.h"
#include "ch7/ndt_3d.h"
#include "common/lidar_utils.h"
#include "common/point_cloud_utils.h"
#include "common/sys_utils.h"

DEFINE_string(source, "/home/holo/Downloads/hyper_matching/old_map.ply", "第1个点云路径");
DEFINE_string(target, "/home/holo/Downloads/hyper_matching/new_map.ply", "第2个点云路径");

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::CloudPtr source(new sad::PointCloudType), target(new sad::PointCloudType);
    pcl::io::loadPLYFile(fLS::FLAGS_source, *source);
    // pcl::io::loadPLYFile(fLS::FLAGS_target, *target);
    LOG(INFO) << "load source and target over";

    /*
    source = sad::VoxelCloud(source, 0.5);
    target = sad::VoxelCloud(target, 0.5);
    pcl::io::savePLYFileASCII("/home/holo/Downloads/hyper_matching/voxel_old.ply", *source);
    pcl::io::savePLYFileASCII("/home/holo/Downloads/hyper_matching/voxel_new.ply", *target);
    */

    SE3 pose;
    SO3 R = SO3::rotZ(M_PI / 2.0);
    pose.setRotationMatrix(R.matrix());

    Eigen::Matrix3d RR;
    RR << 0.999139487743, -0.041377406567, 0.003021436278, 0.041374932975, 0.999143421650, 0.000883641071,
        -0.003055412322, -0.000757867470, 0.99999505281;
    Eigen::Quaterniond q(RR);
    LOG(INFO) << "raw R" << q.toRotationMatrix();
    q.normalize();
    LOG(INFO) << "normalized R" << q.toRotationMatrix();
    Eigen::Vector3d tt;
    tt << 2.734453916550, -3.578369617462, -0.912014007568;
    SE3 pose2(q, tt);
    pose2 *= pose;

    sad::CloudPtr source_trans(new sad::PointCloudType);
    pcl::transformPointCloud(*source, *source_trans, pose2.matrix().cast<float>());
    const std::string filePath = "/home/holo/Downloads/hyper_matching/init_hyper_trans.ply";
    source_trans->height       = 1;
    source_trans->width        = source_trans->size();
    pcl::io::savePLYFileASCII(filePath, *source_trans);
    LOG(INFO) << "save transformed cloud";
}
