//
// Created by xiang on 22-12-20.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "ch7/ndt_3d.h"
#include "common/eigen_types.h"
#include "common/point_cloud_utils.h"
#include "keyframe.h"

DEFINE_string(map_path, "./data/ch9/", "导出数据的目录");
DEFINE_double(voxel_size, 0.1, "导出地图分辨率");

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;

    std::map<IdType, KFPtr> keyframes;
    if (!LoadKeyFrames("./data/ch9/keyframes.txt", keyframes))
    {
        LOG(ERROR) << "failed to load keyframes";
        return 0;
    }

    std::map<Vec2i, CloudPtr, less_vec<2>> map_data;  // 以网格ID为索引的地图数据
    pcl::VoxelGrid<PointType>              voxel_grid_filter;
    float                                  resolution = FLAGS_voxel_size;
    voxel_grid_filter.setLeafSize(resolution, resolution, resolution);

    // 逻辑和dump map差不多，但每个点个查找它的网格ID，没有的话会创建
    for (auto& kfp : keyframes)
    {
        auto kf = kfp.second;
        kf->LoadScan("./data/ch9/");

        CloudPtr cloud_trans(new PointCloudType);
        pcl::transformPointCloud(*kf->cloud_, *cloud_trans, kf->opti_pose_2_.matrix());

        // voxel size
        CloudPtr kf_cloud_voxeled(new PointCloudType);
        voxel_grid_filter.setInputCloud(cloud_trans);
        voxel_grid_filter.filter(*kf_cloud_voxeled);

        LOG(INFO) << "building kf " << kf->id_ << " in " << keyframes.size();

        // add to grid
        for (const auto& pt : kf_cloud_voxeled->points)
        {
            int   gx = floor((pt.x - 50.0) / 100);
            int   gy = floor((pt.y - 50.0) / 100);
            Vec2i key(gx, gy);
            auto  iter = map_data.find(key);
            if (iter == map_data.end())
            {
                // create point cloud
                CloudPtr cloud(new PointCloudType);
                cloud->points.emplace_back(pt);
                cloud->is_dense = false;
                cloud->height   = 1;
                map_data.emplace(key, cloud);
            }
            else
            {
                iter->second->points.emplace_back(pt);
            }
        }
    }

    // 存储点云和索引文件
    LOG(INFO) << "saving maps, grids: " << map_data.size();
    std::system("mkdir -p ./data/ch9/map_data/");
    std::system("rm -rf ./data/ch9/map_data/*");  // 清理一下文件夹
    std::ofstream fout("./data/ch9/map_data/map_index.txt");

    std::map<Vec2i, Ndt3d::NdtGrid, less_vec<2>> grid_map_data;  // 以网格ID为索引的地图数据
    for (auto& dp : map_data)
    {
        // 生成ndt grid
        Ndt3d::Options ndt_option;
        ndt_option.voxel_size_        = 1;
        ndt_option.min_effective_pts_ = 5;
        Ndt3d          ndt_3d(ndt_option);
        Ndt3d::NdtGrid ndt_grid = ndt_3d.BuildVoxels(dp.second);
        if (ndt_grid.size() != 0)
        {
            grid_map_data.emplace(dp.first, ndt_grid);
        }
        LOG(INFO) << "ndt_grid info : " << ndt_grid.size();

        fout << dp.first[0] << " " << dp.first[1] << std::endl;
        dp.second->width = dp.second->size();
        sad::VoxelGrid(dp.second, 0.1);

        sad::SaveCloudToFile("./data/ch9/map_data/" + std::to_string(dp.first[0]) + "_" + std::to_string(dp.first[1]) +
                                 ".pcd",
                             *dp.second);
        LOG(INFO) << "dp second size : " << dp.second->size();
    }
    fout.close();

    std::ofstream fout2("./data/ch9/map_data/ndt_map_index.txt");
    for (const auto& dp : grid_map_data)
    {
        fout2 << dp.first[0] << " " << dp.first[1] << std::endl;
        Ndt3d::WriteNdtGridToFile(dp.second, "./data/ch9/map_data/" + std::to_string(dp.first[0]) + "_" +
                                                 std::to_string(dp.first[1]) + ".ndt");
    }
    fout2.close();

    LOG(INFO) << "done.";
    return 0;
}
