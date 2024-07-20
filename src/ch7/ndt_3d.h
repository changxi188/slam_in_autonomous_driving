//
// Created by xiang on 2022/7/14.
//

#ifndef SLAM_IN_AUTO_DRIVING_NDT_3D_H
#define SLAM_IN_AUTO_DRIVING_NDT_3D_H

#include <fstream>
#include "common/eigen_types.h"
#include "common/point_types.h"

namespace sad
{
/**
 * 3D 形式的NDT
 */
class Ndt3d
{
public:
    enum class NearbyType
    {
        CENTER,   // 只考虑中心
        NEARBY6,  // 上下左右前后
    };

    struct Options
    {
        int    max_iteration_     = 20;     // 最大迭代次数
        double voxel_size_        = 1.0;    // 体素大小
        double inv_voxel_size_    = 1.0;    //
        int    min_effective_pts_ = 10;     // 最近邻点数阈值
        int    min_pts_in_voxel_  = 3;      // 每个栅格中最小点数
        double eps_               = 1e-2;   // 收敛判定条件
        double res_outlier_th_    = 20.0;   // 异常值拒绝阈值
        bool   remove_centroid_   = false;  // 是否计算两个点云中心并移除中心？

        NearbyType nearby_type_ = NearbyType::NEARBY6;
    };
    using KeyType = Eigen::Matrix<int, 3, 1>;  // 体素的索引

    struct VoxelData
    {
        VoxelData()
        {
        }
        VoxelData(size_t id)
        {
            idx_.emplace_back(id);
        }

        std::vector<size_t> idx_;                    // 点云中点的索引
        Vec3d               mu_    = Vec3d::Zero();  // 均值
        Mat3d               sigma_ = Mat3d::Zero();  // 协方差
        Mat3d               info_  = Mat3d::Zero();  // 协方差之逆

        void WriteToFile(std::ofstream& out) const
        {
            // Write vector size and elements
            size_t idx_size = idx_.size();
            out.write(reinterpret_cast<const char*>(&idx_size), sizeof(idx_size));
            out.write(reinterpret_cast<const char*>(idx_.data()), idx_size * sizeof(size_t));

            // Write Vec3d mu_
            out.write(reinterpret_cast<const char*>(mu_.data()), mu_.size() * sizeof(double));

            // Write Mat3d sigma_
            out.write(reinterpret_cast<const char*>(sigma_.data()), sigma_.size() * sizeof(double));

            // Write Mat3d info_
            out.write(reinterpret_cast<const char*>(info_.data()), info_.size() * sizeof(double));
        }

        void ReadFromFile(std::ifstream& in)
        {
            // Read vector size and elements
            size_t idx_size;
            in.read(reinterpret_cast<char*>(&idx_size), sizeof(idx_size));
            idx_.resize(idx_size);
            in.read(reinterpret_cast<char*>(idx_.data()), idx_size * sizeof(size_t));

            // Read Vec3d mu_
            in.read(reinterpret_cast<char*>(mu_.data()), mu_.size() * sizeof(double));

            // Read Mat3d sigma_
            in.read(reinterpret_cast<char*>(sigma_.data()), sigma_.size() * sizeof(double));

            // Read Mat3d info_
            in.read(reinterpret_cast<char*>(info_.data()), info_.size() * sizeof(double));
        }
    };
    using NdtGrid = std::unordered_map<KeyType, VoxelData, hash_vec<3>>;

    static void WriteNdtGridToFile(const NdtGrid& grid, const std::string& filename)
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
        {
            throw std::ios_base::failure("Failed to open file for writing");
        }

        // Write the number of elements in the map
        size_t map_size = grid.size();
        out.write(reinterpret_cast<const char*>(&map_size), sizeof(map_size));

        for (const auto& pair : grid)
        {
            // Write the key
            out.write(reinterpret_cast<const char*>(pair.first.data()), pair.first.size() * sizeof(size_t));

            // Write the value
            pair.second.WriteToFile(out);
        }

        out.close();
    }

    static void ReadNdtGridFromFile(NdtGrid& grid, const std::string& filename)
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
        {
            throw std::ios_base::failure("Failed to open file for reading");
        }

        // Read the number of elements in the map
        size_t map_size;
        in.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));

        for (size_t i = 0; i < map_size; ++i)
        {
            // Read the key
            KeyType key;
            in.read(reinterpret_cast<char*>(key.data()), key.size() * sizeof(size_t));

            // Read the value
            VoxelData value;
            value.ReadFromFile(in);

            grid[key] = value;
        }

        in.close();
    }

    Ndt3d()
    {
        options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
        GenerateNearbyGrids();
    }

    Ndt3d(Options options) : options_(options)
    {
        options_.inv_voxel_size_ = 1.0 / options_.voxel_size_;
        GenerateNearbyGrids();
    }

    /// 设置目标的Scan
    void SetTarget(CloudPtr target)
    {
        target_ = target;
        BuildVoxels();

        // 计算点云中心
        target_center_ = std::accumulate(target->points.begin(), target_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         target_->size();
    }

    /// 设置被配准的Scan
    void SetSource(CloudPtr source)
    {
        source_ = source;

        source_center_ = std::accumulate(source_->points.begin(), source_->points.end(), Vec3d::Zero().eval(),
                                         [](const Vec3d& c, const PointType& pt) -> Vec3d { return c + ToVec3d(pt); }) /
                         source_->size();
    }

    void SetGtPose(const SE3& gt_pose)
    {
        gt_pose_ = gt_pose;
        gt_set_  = true;
    }

    /// 使用gauss-newton方法进行ndt配准
    bool AlignNdt(SE3& init_pose);

    double GetScore()
    {
        return score_;
    }

    NdtGrid BuildVoxels(const CloudPtr cloud);

private:
    void BuildVoxels();

    /// 根据最近邻的类型，生成附近网格
    void GenerateNearbyGrids();

    CloudPtr target_ = nullptr;
    CloudPtr source_ = nullptr;

    Vec3d target_center_ = Vec3d::Zero();
    Vec3d source_center_ = Vec3d::Zero();

    SE3  gt_pose_;
    bool gt_set_ = false;

    Options options_;

    NdtGrid              grids_;         // 栅格数据
    std::vector<KeyType> nearby_grids_;  // 附近的栅格

    double score_ = 0;
};

}  // namespace sad

#endif  // SLAM_IN_AUTO_DRIVING_NDT_3D_H
