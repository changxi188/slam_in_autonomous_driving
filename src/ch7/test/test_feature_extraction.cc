//
// Created by xiang on 2022/7/18.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ch7/loam-like/feature_extraction.h"
#include "common/io_utils.h"

#include "common/point_cloud_utils.h"
#include "common/timer/timer.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <pcl/visualization/pcl_visualizer.h>

/// 这里需要vlp16的数据，用wxb的
DEFINE_string(bag_path, "./dataset/sad/wxb/test1.bag", "path to wxb bag");

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    // 测试角点和平面点的提取
    sad::FeatureExtraction feature_extraction;

    // system("rm -rf ./data/ch7/*.pcd");

    pcl::visualization::PCLVisualizer::Ptr viewer_(new pcl::visualization::PCLVisualizer());

    int  i               = 0;
    auto velodyne_handle = [&](sad::FullCloudPtr cloud) -> bool {
        sad::FullCloudPtr valid_pc(new sad::FullPointCloudType);
        sad::CloudPtr     pcd_corner(new sad::PointCloudType), pcd_surf(new sad::PointCloudType);
        sad::common::Timer::Evaluate([&]() { feature_extraction.Extract(cloud, pcd_corner, pcd_surf, valid_pc); },
                                     "Feature Extraction");
        LOG(INFO) << "original pts:" << cloud->size() << ", corners: " << pcd_corner->size()
                  << ", surf: " << pcd_surf->size();

        viewer_->removePointCloud("corner");
        viewer_->removePointCloud("surf");
        pcl::visualization::PointCloudColorHandlerCustom<sad::PointType> corner_field_color(pcd_corner, 0, 255, 0);
        viewer_->addPointCloud<sad::PointType>(pcd_corner, corner_field_color, "corner");
        pcl::visualization::PointCloudColorHandlerCustom<sad::PointType> surf_field_color(pcd_surf, 255, 0, 0);
        viewer_->addPointCloud<sad::PointType>(pcd_surf, surf_field_color, "surf");

        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        viewer_->spinOnce(1);
        /*
        sad::SaveCloudToFile("./data/ch7/fd/" + std::to_string(i) + "corner.pcd", *pcd_corner);
        sad::SaveCloudToFile("./data/ch7/fd/" + std::to_string(i++) + "surf.pcd", *pcd_surf);
        */
        return true;
    };

    sad::RosbagIO bag_io(fLS::FLAGS_bag_path);
    bag_io.AddVelodyneHandle("/velodyne_packets_1", velodyne_handle).Go();

    sad::common::Timer::PrintAll();
    LOG(INFO) << "done.";

    return 0;
}
