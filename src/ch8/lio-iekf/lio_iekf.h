#ifndef SAD_CH8_LASER_MAPPING_H
#define SAD_CH8_LASER_MAPPING_H

#include <livox_ros_driver/CustomMsg.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>

/// 部分类直接使用ch7的结果
#include "ch3/static_imu_init.h"
#include "ch7/loosely_coupled_lio/cloud_convert.h"
#include "ch7/loosely_coupled_lio/measure_sync.h"
#include "ch7/ndt_inc.h"
#include "ch8/ikd-Tree/ikd_Tree.h"
#include "ch8/lio-iekf/iekf.hpp"
#include "tools/ui/pangolin_window.h"

namespace sad
{
#define SKEW_SYM_MATRX(v) 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define LASER_POINT_COV (0.1)
class LioIEKF
{
public:
    const static int NUM_MATCH_POINTS = 5;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    struct Options
    {
        Options()
        {
        }
        bool save_motion_undistortion_pcd_ = false;  // 是否保存去畸变前后的点云
        bool with_ui_                      = true;   // 是否带着UI
    };

    LioIEKF(Options options = Options());
    ~LioIEKF() = default;

    /// init without ros
    bool Init(const std::string& config_yaml);

    /// 点云回调函数
    void PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg);

    /// IMU回调函数
    void IMUCallBack(IMUPtr msg_in);

    /// 结束程序，退出UI
    void Finish();

    /// 获取当前姿态
    NavStated GetCurrentState() const
    {
        return ieskf_.GetNominalState();
    }

    /// 获取当前扫描
    CloudPtr GetCurrentScan() const
    {
        return current_scan_;
    }

private:
    bool LoadFromYAML(const std::string& yaml_file);

    /// 处理同步之后的IMU和雷达数据
    void ProcessMeasurements(const MeasureGroup& meas);

    /// 尝试让IMU初始化
    void TryInitIMU();

    /// 利用IMU预测状态信息
    /// 这段时间的预测数据会放入imu_states_里
    void Predict();

    /// 对measures_中的点云去畸变
    void Undistort();

    /// 执行一次配准和观测
    void Align();

    void LasermapFovSegment();

    void MapIncremental(const CloudPtr feats_down_body, const SE3& current_pose,
                        const vector<PointVec>& Nearest_Points);

    void IKdtreeComputeResidualAndJacobians(const CloudPtr pointcloud, const SE3& input_pose, Mat18d& HTVH,
                                            Vec18d& HTVr, vector<PointVec>& Nearest_Points);

    /// modules
    std::shared_ptr<MessageSync> sync_ = nullptr;
    StaticIMUInit                imu_init_;

    /// point clouds data
    FullCloudPtr scan_undistort_{new FullPointCloudType()};  // scan after undistortion
    CloudPtr     current_scan_ = nullptr;

    /// NDT数据
    IncNdt3d ndt_;
    SE3      last_pose_;

    // flags
    bool imu_need_init_  = true;
    bool flg_first_scan_ = true;
    int  frame_num_      = 0;

    ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
    MeasureGroup           measures_;  // sync IMU and lidar scan
    std::vector<NavStated> imu_states_;
    IESKFD                 ieskf_;  // IESKF
    SE3                    TIL_;    // Lidar与IMU之间外参

    Options                             options_;
    std::shared_ptr<ui::PangolinWindow> ui_ = nullptr;

    double                      filter_size_map_min = 0.5;
    IKDTREE::KD_TREE<PointType> ikdtree;

    vector<IKDTREE::BoxPointType> cub_needrm;
    int                           kdtree_delete_counter = 0;
    bool                          Localmap_Initialized  = false;  // 局部地图是否初始化
    IKDTREE::BoxPointType         LocalMap_Points;                // ikd-tree地图立方体的2个角点
    double                        cube_len      = 1000;
    const float                   MOV_THRESHOLD = 1.5f;
    float                         DET_RANGE     = 300.0f;  // 激光雷达的最大探测范围
    // int                           feats_down_size = 0;
};

}  // namespace sad

#endif  // FASTER_LIO_LASER_MAPPING_H
