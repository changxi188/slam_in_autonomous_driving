
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

DEFINE_string(map_path, "./data/ch9/map_data/", "导出数据的目录");
DEFINE_double(voxel_size, 0.1, "导出地图分辨率");

void LoadNdtMapIndex(const std::string& data_path, std::set<Vec2i, sad::less_vec<2>>& ndt_map_data_index)
{
    std::ifstream fin(data_path + "/ndt_map_index.txt");
    while (!fin.eof())
    {
        int x, y;
        fin >> x >> y;
        ndt_map_data_index.emplace(Vec2i(x, y));
    }
    fin.close();
}

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold  = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags(&argc, &argv, true);

    using namespace sad;

    std::set<Vec2i, sad::less_vec<2>> ndt_map_data_index;  // 哪些格子存在地图数据
    LoadNdtMapIndex(FLAGS_map_path, ndt_map_data_index);

    for (const auto& index : ndt_map_data_index)
    {
        LOG(INFO) << "index : " << index.x() << " " << index.y();
        Ndt3d::NdtGrid grids;  // 栅格数据
                               //
        std::string ndt_file = FLAGS_map_path + std::to_string(index.x()) + "_" + std::to_string(index.y()) + ".ndt";
        Ndt3d::ReadNdtGridFromFile(grids, ndt_file);
        LOG(INFO) << "grid info : " << grids.size();
    }
}
