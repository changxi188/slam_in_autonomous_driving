//
// Created by changxi on 2023/11/27.
//
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/highgui.hpp>

#include "ch6/lidar_2d_utils.h"
#include "common/io_utils.h"
#include "common/math_utils.h"

#include <laser_geometry/laser_geometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include "DBSCAN_kdtree.h"
#include "DBSCAN_simple.h"

DEFINE_string(bag_path, "./dataset/sad/2dmapping/floor4.bag", "数据包路径");

// 将Scan转换为PointCloud
void ConvertFromScanToPointCloud(Scan2d::Ptr scan, pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud)
{
    for (size_t i = 0; i < scan->ranges.size(); ++i)
    {
        if (scan->ranges[i] < scan->range_min || scan->ranges[i] > scan->range_max)
        {
            continue;
        }

        double real_angle = scan->angle_min + i * scan->angle_increment;
        double x          = scan->ranges[i] * std::cos(real_angle);
        double y          = scan->ranges[i] * std::sin(real_angle);

        if (real_angle < scan->angle_min + 30 * M_PI / 180.0 || real_angle > scan->angle_max - 30 * M_PI / 180.0)
        {
            continue;
        }
        pcl::PointXYZ pt;
        pt.x = x;
        pt.y = y;
        pt.z = 0;
        raw_cloud->push_back(pt);
    }
}

// 运行DBSCAN
std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> RunDBSCAN(const pcl::PointCloud<pcl::PointXYZ>::Ptr& raw_cloud,
                                                            const double                               epsilon)
{
    std::cout << "-------------------------- DBSCAN Algorithm ------------------------------------- " << std::endl;
    std::cout << "begin construct kdtree" << std::endl;
    std::chrono::steady_clock::time_point   t1 = std::chrono::steady_clock::now();
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(raw_cloud);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used  = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "construct KDTree over, cost : " << time_used.count() << " second. " << std::endl;

    std::cout << "begin run DBSCAN " << std::endl;
    t1 = std::chrono::steady_clock::now();
    std::vector<pcl::PointIndices>     cluster_indices;
    DBSCANKdtreeCluster<pcl::PointXYZ> ec;
    ec.setCorePointMinPts(5);
    ec.setClusterTolerance(epsilon);
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(2500);
    ec.setSearchMethod(tree);
    ec.setInputCloud(raw_cloud);
    ec.extract(cluster_indices);
    t2        = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "DBSCAN over, cost : " << time_used.count() << " second, cluster number : " << cluster_indices.size()
              << std::endl;

    std::cout << "begin construct return value " << std::endl;
    t1 = std::chrono::steady_clock::now();
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_clusters;
    int                                               j = 0;

    // visualization, use indensity to show different color for each cluster.
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end();
         it++, j++)
    {
        float                                intensity = j * 10;
        pcl::PointCloud<pcl::PointXYZI>::Ptr new_clustered(new pcl::PointCloud<pcl::PointXYZI>);
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
        {
            pcl::PointXYZI tmp;
            tmp.x         = raw_cloud->points[*pit].x;
            tmp.y         = raw_cloud->points[*pit].y;
            tmp.z         = raw_cloud->points[*pit].z;
            tmp.intensity = 0.1 + intensity;
            new_clustered->points.push_back(tmp);
        }
        new_clustered->width  = new_clustered->points.size();
        new_clustered->height = 1;

        cloud_clusters.push_back(new_clustered);
    }
    t2        = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "construct return value over, cost : " << time_used.count() << " second." << std::endl;

    return cloud_clusters;
}

// 可视化
void Visualization(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cloud_clusters,
                   pcl::visualization::CloudViewer& viewer, Scan2d::Ptr scan)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr all_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (const auto cloud : cloud_clusters)
    {
        *all_cloud += *cloud;
    }
    viewer.showCloud(all_cloud);
    cv::Mat image;
    sad::Visualize2DScan(scan, SE2(), image, Vec3b(255, 0, 0));
    cv::imshow("scan", image);
    cv::waitKey(0);
}

// 计算两条直线的夹角
double calculateAngle(const Vec3d& coeff1, const Vec3d& coeff2)
{
    // 计算斜率
    double m1 = -coeff1[0] / coeff1[1];
    double m2 = -coeff2[0] / coeff2[1];

    // 计算夹角的切线斜率
    double tanTheta = fabs((m2 - m1) / (1 + m1 * m2));

    // 计算夹角的弧度
    double angleRad = atan(tanTheta);

    // 将弧度转换为度数
    double angleDeg = angleRad * 180 / M_PI;

    return angleDeg;
}

// 通过直线拟合做退化判定
bool DegradationDection(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cloud_clusters)
{
    std::vector<Vec3d> line_coeffs;
    for (const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud : cloud_clusters)
    {
        std::vector<Vec2d> effective_pts;  // 有效点
        for (size_t i = 0; i < cloud->points.size(); ++i)
        {
            Vec2d pt;
            pt[0] = cloud->points[i].x;
            pt[1] = cloud->points[i].y;
            effective_pts.emplace_back(pt);
        }

        Vec3d line_coeff;
        sad::math::FitLine2D(effective_pts, line_coeff);
        line_coeffs.push_back(line_coeff);
    }

    bool   has_big_angle = false;
    double max_angle     = 0.0;
    for (size_t i = 0; i < line_coeffs.size() - 1; ++i)
    {
        for (size_t j = i + 1; j < line_coeffs.size(); ++j)
        {
            Vec3d  coeff1 = line_coeffs.at(i);
            Vec3d  coeff2 = line_coeffs.at(j);
            double angle  = calculateAngle(coeff1, coeff2);
            if (angle > max_angle)
            {
                max_angle = angle;
            }
        }
    }

    // 如果最大角度大于5.5度就认为没有退化
    if (max_angle >= 5.5)
    {
        LOG(INFO) << "No degradation detected ,max_angle :" << max_angle;
        return false;
    }

    LOG(INFO) << "Degradation detected ,max_angle :" << max_angle;

    return true;
}

/// 测试从rosbag中读取2d scan并plot的结果
int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    pcl::visualization::CloudViewer viewer("aa");

    auto degradation_detection = [&](Scan2d::Ptr scan) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);

        // step1. 将Scan转换为PointCloud
        ConvertFromScanToPointCloud(scan, raw_cloud);

        // step2. 利用dbscan做聚类
        std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cloud_clusters = RunDBSCAN(raw_cloud, 0.2);

        // step3. 通过直线拟合做退化判定
        bool is_degration = DegradationDection(cloud_clusters);

        // 可视化聚类结果
        Visualization(cloud_clusters, viewer, scan);

        return is_degration;
    };

    sad::RosbagIO rosbag_io(fLS::FLAGS_bag_path);
    rosbag_io.AddScan2DHandle("pavo_scan_bottom", degradation_detection).Go();

    while (!viewer.wasStopped())
    {
    }

    return 0;
}
