#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ch7/icp_3d.h"
#include "common/point_cloud_utils.h"

DEFINE_string(source, "/home/cheng/Downloads/icp_test/1671797384.4706228_luowa.ply", "第1个点云路径");
DEFINE_string(target, "/home/cheng/Downloads/icp_test/all(4).ply", "第2个点云路径");

DEFINE_string(voxel_source, "/home/cheng/Downloads/icp_test/voxel_source.ply", "第1个点云路径");
DEFINE_string(voxel_target, "/home/cheng/Downloads/icp_test/voxel_target.ply", "第2个点云路径");

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    sad::CloudPtr source(new sad::PointCloudType), target(new sad::PointCloudType);
    pcl::io::loadPLYFile(fLS::FLAGS_voxel_source, *source);
    pcl::io::loadPLYFile(fLS::FLAGS_voxel_target, *target);

    SO3 R = SO3::rotZ(M_PI / 2.0);
    SE3 T_init(R, Vec3d());

    pcl::transformPointCloud(*source, *source, T_init.matrix().cast<float>());

    sad::Icp3d::Options options;
    options.max_iteration_           = 20;
    options.max_nn_distance_         = 5.0;
    options.max_plane_distance_      = 5.0;
    options.max_line_distance_       = 0.5;
    options.min_effective_pts_       = 100;
    options.eps_                     = 1e-2;
    options.use_initial_translation_ = true;

    sad::Icp3d icp(options);
    icp.SetSource(source);
    icp.SetTarget(target);
    SE3  pose;
    bool success = icp.AlignP2Plane(pose);
    if (success)
    {
        LOG(INFO) << "icp p2plane align success, pose: " << pose.so3().unit_quaternion().coeffs().transpose() << ", "
                  << pose.translation().transpose();
        LOG(INFO) << "Pose : " << pose.matrix();
        pcl::transformPointCloud(*source, *source, pose.matrix().cast<float>());
    }
    else
    {
        LOG(ERROR) << "align failed.";
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer());

    sad::RemoveGround(source, -1.2);
    // sad::RemoveGround(target, -1.0);

    pcl::visualization::PointCloudColorHandlerGenericField<sad::PointType> source_fieldColor(source, "z");
    viewer->addPointCloud<sad::PointType>(source, source_fieldColor, "source");

    pcl::visualization::PointCloudColorHandlerGenericField<sad::PointType> target_fieldColor(target, "z");
    viewer->addPointCloud<sad::PointType>(target, target_fieldColor, "target");

    while (true)
    {
        viewer->spinOnce(1);
    }

    /*
    sad::VoxelGrid(source, 0.5);
    sad::VoxelGrid(target, 0.5);

    pcl::io::savePLYFileASCII(fLS::FLAGS_voxel_source, *source);
    pcl::io::savePLYFileASCII(fLS::FLAGS_voxel_target, *target);
    */
}
