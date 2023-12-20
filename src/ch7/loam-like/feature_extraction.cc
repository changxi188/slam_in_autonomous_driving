//
// Created by xiang on 2022/7/21.
//
#include <glog/logging.h>
#include "ch7/loam-like/feature_extraction.h"
#include "tools/pointcloud_convert/packets_parser.h"

namespace sad
{
void FeatureExtraction::Extract(FullCloudPtr pc_in, CloudPtr pc_out_edge, CloudPtr pc_out_surf)
{
    std::vector<CloudPtr> scans_in_each_line;  // 分线数的点云
    for (int i = 0; i < num_scans_; i++)
    {
        scans_in_each_line.emplace_back(new PointCloudType);
    }

    for (auto& pt : pc_in->points)
    {
        assert(pt.ring >= 0 && pt.ring < num_scans_);

        PointType p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;

        if (tools::PacketsParser::isPointValid(pt))
        {
            p.intensity = pt.intensity;
        }
        else
        {
            p.intensity = -1;
        }

        scans_in_each_line[pt.ring]->points.emplace_back(p);
    }

    // 提取地面
    std::size_t horizen_size = scans_in_each_line[0]->size();
    float       diff_x, diff_y, diff_z, angle;
    ground_mat_ = cv::Mat(num_scans_, horizen_size, CV_8S, cv::Scalar::all(0));
    for (std::size_t j = 0; j < horizen_size; ++j)
    {
        for (std::size_t i = 0; i < ground_scan_ind_; ++i)
        {
            PointType lower_point = scans_in_each_line[i]->at(j);
            PointType upper_point = scans_in_each_line[i + 1]->at(j);

            if (lower_point.intensity == -1 || upper_point.intensity == -1)
            {
                ground_mat_.at<int8_t>(i, j) = -1;
                continue;
            }

            // 由上下两线之间点的XYZ位置得到两线之间的俯仰角
            // 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
            // 否则，则不是地面点，进行后续操作
            diff_x = upper_point.x - lower_point.x;
            diff_y = upper_point.y - lower_point.y;
            diff_z = upper_point.z - lower_point.z;

            angle = atan2(diff_z, sqrt(diff_x * diff_x + diff_y * diff_y)) * 180 / M_PI;

            if (abs(angle - sensor_mount_angle_) <= 10)
            {
                ground_mat_.at<int8_t>(i, j)     = 1;
                ground_mat_.at<int8_t>(i + 1, j) = 1;
            }
        }
    }

    // 处理曲率
    for (int i = 0; i < num_scans_; i++)
    {
        if (scans_in_each_line[i]->size() != 1800 && scans_in_each_line[i]->size() != 1824)
        {
            LOG(INFO) << "size:" << scans_in_each_line[i]->size();
        }
        /*
        if (scans_in_each_line[i]->points.size() < 131)
        {
            continue;
        }
        */
        // assert(scans_in_each_line[i]->size() == 1800 || scans_in_each_line[i]->size() == 1824);

        std::vector<IdAndValue> cloud_curvature;  // 每条线对应的曲率
        int                     total_points = scans_in_each_line[i]->points.size() - 10;
        for (int j = 5; j < (int)scans_in_each_line[i]->points.size() - 5; j++)
        {
            // 两头留一定余量，采样周围10个点取平均值
            double diffX = scans_in_each_line[i]->points[j - 5].x + scans_in_each_line[i]->points[j - 4].x +
                           scans_in_each_line[i]->points[j - 3].x + scans_in_each_line[i]->points[j - 2].x +
                           scans_in_each_line[i]->points[j - 1].x - 10 * scans_in_each_line[i]->points[j].x +
                           scans_in_each_line[i]->points[j + 1].x + scans_in_each_line[i]->points[j + 2].x +
                           scans_in_each_line[i]->points[j + 3].x + scans_in_each_line[i]->points[j + 4].x +
                           scans_in_each_line[i]->points[j + 5].x;
            double diffY = scans_in_each_line[i]->points[j - 5].y + scans_in_each_line[i]->points[j - 4].y +
                           scans_in_each_line[i]->points[j - 3].y + scans_in_each_line[i]->points[j - 2].y +
                           scans_in_each_line[i]->points[j - 1].y - 10 * scans_in_each_line[i]->points[j].y +
                           scans_in_each_line[i]->points[j + 1].y + scans_in_each_line[i]->points[j + 2].y +
                           scans_in_each_line[i]->points[j + 3].y + scans_in_each_line[i]->points[j + 4].y +
                           scans_in_each_line[i]->points[j + 5].y;
            double diffZ = scans_in_each_line[i]->points[j - 5].z + scans_in_each_line[i]->points[j - 4].z +
                           scans_in_each_line[i]->points[j - 3].z + scans_in_each_line[i]->points[j - 2].z +
                           scans_in_each_line[i]->points[j - 1].z - 10 * scans_in_each_line[i]->points[j].z +
                           scans_in_each_line[i]->points[j + 1].z + scans_in_each_line[i]->points[j + 2].z +
                           scans_in_each_line[i]->points[j + 3].z + scans_in_each_line[i]->points[j + 4].z +
                           scans_in_each_line[i]->points[j + 5].z;
            IdAndValue distance(j, diffX * diffX + diffY * diffY + diffZ * diffZ);
            cloud_curvature.push_back(distance);
        }

        // 对每个区间选取特征，把360度分为6个区间
        for (int j = 0; j < 6; j++)
        {
            int sector_length = (int)(total_points / 6);
            int sector_start  = sector_length * j;
            int sector_end    = sector_length * (j + 1) - 1;
            if (j == 5)
            {
                sector_end = total_points - 1;
            }

            std::vector<IdAndValue> sub_cloud_curvature(cloud_curvature.begin() + sector_start,
                                                        cloud_curvature.begin() + sector_end);

            ExtractFromSector(scans_in_each_line[i], sub_cloud_curvature, pc_out_edge, pc_out_surf);
        }
    }
}

void FeatureExtraction::ExtractFromSector(const CloudPtr& pc_in, std::vector<IdAndValue>& cloud_curvature,
                                          CloudPtr& pc_out_edge, CloudPtr& pc_out_surf)
{
    // 按曲率排序
    std::sort(cloud_curvature.begin(), cloud_curvature.end(),
              [](const IdAndValue& a, const IdAndValue& b) { return a.value_ < b.value_; });

    int largest_picked_num = 0;
    int point_info_count   = 0;

    /// 按照曲率最大的开始搜，选取曲率最大的角点
    std::vector<int> picked_points;  // 标记被选中的角点，角点附近的点都不会被选取
    for (int i = cloud_curvature.size() - 1; i >= 0; i--)
    {
        int ind = cloud_curvature[i].id_;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
        {
            if (cloud_curvature[i].value_ <= 0.1)
            {
                break;
            }

            largest_picked_num++;
            picked_points.push_back(ind);

            if (largest_picked_num <= 20)
            {
                pc_out_edge->push_back(pc_in->points[ind]);
                point_info_count++;
            }
            else
            {
                break;
            }

            for (int k = 1; k <= 5; k++)
            {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k - 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k - 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k - 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }
                picked_points.push_back(ind + k);
            }
            for (int k = -1; k >= -5; k--)
            {
                double diffX = pc_in->points[ind + k].x - pc_in->points[ind + k + 1].x;
                double diffY = pc_in->points[ind + k].y - pc_in->points[ind + k + 1].y;
                double diffZ = pc_in->points[ind + k].z - pc_in->points[ind + k + 1].z;
                if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                {
                    break;
                }
                picked_points.push_back(ind + k);
            }
        }
    }

    /// 选取曲率较小的平面点
    for (int i = 0; i <= (int)cloud_curvature.size() - 1; i++)
    {
        int ind = cloud_curvature[i].id_;
        if (std::find(picked_points.begin(), picked_points.end(), ind) == picked_points.end())
        {
            pc_out_surf->push_back(pc_in->points[ind]);
        }
    }
}

}  // namespace sad
