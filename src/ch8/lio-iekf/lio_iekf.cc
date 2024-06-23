#include <pcl/common/transforms.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/sparse_block_matrix.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>

// #include "ch4/g2o_types_preinteg.h"
#include "common/g2o_types.h"

#include "common/lidar_utils.h"
#include "common/point_cloud_utils.h"
#include "common/timer/timer.h"

#include "lio_iekf.h"

namespace sad
{
LioIEKF::LioIEKF(Options options) : options_(options)
{
    StaticIMUInit::Options imu_init_options;
    imu_init_options.use_speed_for_static_checking_ = false;  // 本节数据不需要轮速计
    imu_init_                                       = StaticIMUInit(imu_init_options);
}

bool LioIEKF::Init(const std::string& config_yaml)
{
    if (!LoadFromYAML(config_yaml))
    {
        LOG(INFO) << "init failed.";
        return false;
    }

    if (options_.with_ui_)
    {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();
    }

    return true;
}

bool LioIEKF::LoadFromYAML(const std::string& yaml_file)
{
    // get params from yaml
    sync_ = std::make_shared<MessageSync>([this](const MeasureGroup& m) { ProcessMeasurements(m); });
    sync_->Init(yaml_file);

    /// 自身参数主要是雷达与IMU外参
    auto                yaml  = YAML::LoadFile(yaml_file);
    std::vector<double> ext_t = yaml["mapping"]["extrinsic_T"].as<std::vector<double>>();
    std::vector<double> ext_r = yaml["mapping"]["extrinsic_R"].as<std::vector<double>>();

    Vec3d lidar_T_wrt_IMU = math::VecFromArray(ext_t);
    Mat3d lidar_R_wrt_IMU = math::MatFromArray(ext_r);
    TIL_                  = SE3(lidar_R_wrt_IMU, lidar_T_wrt_IMU);
    return true;
}

void LioIEKF::ProcessMeasurements(const MeasureGroup& meas)
{
    LOG(INFO) << "call meas, imu: " << meas.imu_.size() << ", lidar pts: " << meas.lidar_->size();
    measures_ = meas;

    if (imu_need_init_)
    {
        // 初始化IMU系统
        TryInitIMU();
        return;
    }

    // 利用IMU数据进行状态预测
    Predict();

    // 对点云去畸变
    Undistort();

    // 配准
    Align();
}

void LioIEKF::TryInitIMU()
{
    for (auto imu : measures_.imu_)
    {
        imu_init_.AddIMU(*imu);
    }

    if (imu_init_.InitSuccess())
    {
        // 读取初始零偏，设置ESKF
        sad::IESKFD::Options options;
        // 噪声由初始化器估计
        options.gyro_var_ = sqrt(imu_init_.GetCovGyro()[0]);
        options.acce_var_ = sqrt(imu_init_.GetCovAcce()[0]);
        ieskf_.SetInitialConditions(options, imu_init_.GetInitBg(), imu_init_.GetInitBa(), imu_init_.GetGravity());
        imu_need_init_ = false;

        LOG(INFO) << "IMU初始化成功";
    }
}

void LioIEKF::Predict()
{
    imu_states_.clear();
    imu_states_.emplace_back(ieskf_.GetNominalState());

    /// 对IMU状态进行预测
    for (auto& imu : measures_.imu_)
    {
        ieskf_.Predict(*imu);
        imu_states_.emplace_back(ieskf_.GetNominalState());
    }
}

void LioIEKF::Undistort()
{
    auto cloud     = measures_.lidar_;
    auto imu_state = ieskf_.GetNominalState();  // 最后时刻的状态
    SE3  T_end     = SE3(imu_state.R_, imu_state.p_);

    if (options_.save_motion_undistortion_pcd_)
    {
        sad::SaveCloudToFile("./data/ch7/before_undist.pcd", *cloud);
    }

    /// 将所有点转到最后时刻状态上
    std::for_each(std::execution::par_unseq, cloud->points.begin(), cloud->points.end(), [&](auto& pt) {
        SE3       Ti = T_end;
        NavStated match;

        // 根据pt.time查找时间，pt.time是该点打到的时间与雷达开始时间之差，单位为毫秒
        math::PoseInterp<NavStated>(
            measures_.lidar_begin_time_ + pt.time * 1e-3, imu_states_, [](const NavStated& s) { return s.timestamp_; },
            [](const NavStated& s) { return s.GetSE3(); }, Ti, match);

        Vec3d pi           = ToVec3d(pt);
        Vec3d p_compensate = TIL_.inverse() * T_end.inverse() * Ti * TIL_ * pi;

        pt.x = p_compensate(0);
        pt.y = p_compensate(1);
        pt.z = p_compensate(2);
    });
    scan_undistort_ = cloud;

    if (options_.save_motion_undistortion_pcd_)
    {
        sad::SaveCloudToFile("./data/ch7/after_undist.pcd", *cloud);
    }
}

void LioIEKF::LasermapFovSegment()
{
    cub_needrm.clear();  // 清空需要移除的区域
    kdtree_delete_counter = 0;

    SE3 current_pose = ieskf_.GetNominalSE3();

    // W系下位置
    Vec3d pos_LiD = (TIL_.inverse() * current_pose).translation();
    //初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    //各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool  need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;  //如果不需要，直接返回，不更改局部地图

    IKDTREE::BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离
    float mov_dist =
        max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVec points_history;
    ikdtree.acquire_removed_points(points_history);

    if (cub_needrm.size() > 0)
        kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);  //删除指定范围内的点
}

void pointBodyToWorld(PointType const* const pi, PointType* const po, const SE3& pose)
{
    Vec3d p_body(pi->x, pi->y, pi->z);
    Vec3d p_global = pose * p_body;

    po->x         = p_global(0);
    po->y         = p_global(1);
    po->z         = p_global(2);
    po->intensity = pi->intensity;
}

float calc_dist(PointType p1, PointType p2)
{
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

//根据最新估计位姿  增量添加点云到map
void LioIEKF::MapIncremental(const CloudPtr feats_down_body, const SE3& current_pose,
                             const vector<PointVec>& Nearest_Points)
{
    CloudPtr feats_down_world(new PointCloudType);
    feats_down_world->resize(feats_down_body->size());
    int      feats_down_size = feats_down_body->size();
    PointVec PointToAdd;
    PointVec PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        //转换到世界坐标系
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]), current_pose);

        if (!Nearest_Points[i].empty())
        {
            const PointVec&       points_near = Nearest_Points[i];
            bool                  need_add    = true;
            IKDTREE::BoxPointType Box_of_Point;
            PointType             mid_point;  //点所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            float dist = calc_dist(feats_down_world->points[i], mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample.push_back(
                    feats_down_world->points[i]);  //如果距离最近的点都在体素外，则该点不需要Downsample
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[j], mid_point) < dist)  //如果近邻点距离 < 当前点距离，不添加该点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

template <typename T>
bool esti_plane(Eigen::Matrix<T, 4, 1>& pca_result, const PointVec& point, const T& threshold)
{
    Eigen::Matrix<T, LioIEKF::NUM_MATCH_POINTS, 3> A;
    Eigen::Matrix<T, LioIEKF::NUM_MATCH_POINTS, 1> b;
    A.setZero();
    b.setOnes();
    b *= -1.0f;

    //求A/Dx + B/Dy + C/Dz + 1 = 0 的参数
    for (int j = 0; j < LioIEKF::NUM_MATCH_POINTS; j++)
    {
        A(j, 0) = point[j].x;
        A(j, 1) = point[j].y;
        A(j, 2) = point[j].z;
    }

    Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);

    T n = normvec.norm();
    // pca_result是平面方程的4个参数  /n是为了归一化
    pca_result(0) = normvec(0) / n;
    pca_result(1) = normvec(1) / n;
    pca_result(2) = normvec(2) / n;
    pca_result(3) = 1.0 / n;

    //如果几个点中有距离该平面>threshold的点 认为是不好的平面 返回false
    for (int j = 0; j < LioIEKF::NUM_MATCH_POINTS; j++)
    {
        if (fabs(pca_result(0) * point[j].x + pca_result(1) * point[j].y + pca_result(2) * point[j].z + pca_result(3)) >
            threshold)
        {
            return false;
        }
    }
    return true;
}

void LioIEKF::IKdtreeComputeResidualAndJacobians(const CloudPtr feats_down_body, const SE3& input_pose, Mat18d& HTVH,
                                                 Vec18d& HTVr, vector<PointVec>& Nearest_Points)
{
    int feats_down_size = feats_down_body->points.size();

    CloudPtr laserCloudOri(new PointCloudType(100000, 1));  //有效特征点
    CloudPtr corr_normvect(new PointCloudType(100000, 1));  //有效特征点对应点法相量
    CloudPtr normvec(
        new PointCloudType(100000, 1));  //特征点在地图中对应的平面参数(平面的单位法向量,以及当前点到平面距离)
    bool point_selected_surf[100000] = {1};    //判断是否是有效特征点
    for (int i = 0; i < feats_down_size; i++)  //遍历所有的特征点
    {
        PointType& point_body = feats_down_body->points[i];
        PointType  point_world;

        Vec3d p_body(point_body.x, point_body.y, point_body.z);
        //把Lidar坐标系的点先转到IMU坐标系，再根据前向传播估计的位姿x，转到世界坐标系
        Vec3d p_global        = input_pose * p_body;
        point_world.x         = p_global(0);
        point_world.y         = p_global(1);
        point_world.z         = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto&         points_near =
            Nearest_Points[i];  // Nearest_Points[i]打印出来发现是按照离point_world距离，从小到大的顺序的vector

        //寻找point_world的最近邻的平面点
        ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
        //判断是否是有效匹配点，与loam系列类似，要求特征点最近邻的地图点数量>阈值，距离<阈值
        //满足条件的才置为true
        point_selected_surf[i] =
            points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        if (!point_selected_surf[i])
            continue;  //如果该点不满足条件  不进行下面步骤

        Eigen::Matrix<float, 4, 1> pabcd;  //平面点信息
        point_selected_surf[i] = false;    //将该点设置为无效点，用来判断是否满足条件
        //拟合平面方程ax+by+cz+d=0并求解点到平面距离
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z +
                        pabcd(3);  //当前点到平面的距离
            float s =
                1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());  //如果残差大于经验阈值，则认为该点是有效点
                                                            //简言之，距离原点越近的lidar点 要求点到平面的距离越苛刻

            if (s > 0.9)  //如果残差大于阈值，则认为该点是有效点
            {
                point_selected_surf[i] = true;
                normvec->points[i].x   = pabcd(0);  //存储平面的单位法向量  以及当前点到平面距离
                normvec->points[i].y   = pabcd(1);
                normvec->points[i].z   = pabcd(2);
                normvec->points[i].intensity = pd2;
            }
        }

        int effct_feat_num = 0;  //有效特征点的数量
        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_selected_surf[i])  //对于满足要求的点
            {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];  //把这些点重新存到laserCloudOri中
                corr_normvect->points[effct_feat_num] = normvec->points[i];  //存储这些点对应的法向量和到平面的距离
                effct_feat_num++;
            }
        }

        if (effct_feat_num < 1)
        {
            LOG(WARNING) << "No Effective Points!";
            continue;
        }

        // 雅可比矩阵H和残差向量的计算
        Eigen::Matrix<double, Eigen::Dynamic, 9> h_x = Eigen::MatrixXd::Zero(effct_feat_num, 9);
        Eigen::VectorXd                          h   = Eigen::VectorXd::Zero(effct_feat_num);

        for (int i = 0; i < effct_feat_num; i++)
        {
            Vec3d point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
            Mat3d point_crossmat;
            point_crossmat << SKEW_SYM_MATRX(point_);
            Vec3d point_I_ = point_;
            Mat3d point_I_crossmat;
            point_I_crossmat << SKEW_SYM_MATRX(point_I_);

            // 得到对应的平面的法向量
            const PointType& norm_p = corr_normvect->points[i];
            Vec3d            norm_vec(norm_p.x, norm_p.y, norm_p.z);

            // 计算雅可比矩阵H
            Vec3d C(input_pose.so3().matrix().transpose() * norm_vec);
            Vec3d A(point_I_crossmat * C);
            h_x.block<1, 9>(i, 0) << norm_p.x, norm_p.y, norm_p.z, 0.0, 0.0, 0.0, VEC_FROM_ARRAY(A);

            //残差：点面距离
            h(i) = -norm_p.intensity;
        }

        HTVH.setZero();
        HTVr.setZero();
        HTVH.block<9, 9>(0, 0) = (h_x.transpose() * h_x) / LASER_POINT_COV;
        HTVr.block<9, 1>(0, 0) = (h_x.transpose() * h) / LASER_POINT_COV;
    }

    LOG(INFO) << "IKDTREE HTVr : \n" << HTVr;
    LOG(INFO) << "IKDTREE HTVH : \n" << HTVH;
}

void LioIEKF::Align()
{
    FullCloudPtr scan_undistort_trans(new FullPointCloudType);
    pcl::transformPointCloud(*scan_undistort_, *scan_undistort_trans, TIL_.matrix().cast<float>());
    scan_undistort_ = scan_undistort_trans;

    current_scan_ = ConvertToCloud<FullPointType>(scan_undistort_);

    /// the first scan
    if (flg_first_scan_)
    {
        // voxel 之
        pcl::VoxelGrid<PointType> voxel;
        voxel.setLeafSize(0.5, 0.5, 0.5);
        voxel.setInputCloud(current_scan_);

        CloudPtr current_scan_filter(new PointCloudType);
        voxel.filter(*current_scan_filter);

        ndt_.AddCloud(current_scan_);
        flg_first_scan_ = false;

        ikdtree.set_downsample_param(filter_size_map_min);
        ikdtree.Build(current_scan_->points);  //根据世界坐标系下的点构建ikdtree
        return;
    }

    LasermapFovSegment();  //更新localmap边界，然后降采样当前帧点云

    // voxel 之
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(0.5, 0.5, 0.5);
    voxel.setInputCloud(current_scan_);

    CloudPtr current_scan_filter(new PointCloudType);
    voxel.filter(*current_scan_filter);

    // 后续的scan，使用NDT配合pose进行更新
    LOG(INFO) << "=== frame " << frame_num_;

    vector<PointVec> Nearest_Points;
    Nearest_Points.resize(current_scan_filter->size());
    ieskf_.UpdateUsingCustomObserve(
        [this, current_scan_filter, &Nearest_Points](const SE3& input_pose, Mat18d& HTVH, Vec18d& HTVr) {
            // ndt_.SetSource(current_scan_filter);
            // ndt_.ComputeResidualAndJacobians(input_pose, HTVH, HTVr);
            IKdtreeComputeResidualAndJacobians(current_scan_filter, input_pose, HTVH, HTVr, Nearest_Points);
        });

    auto current_nav_state = ieskf_.GetNominalState();

    // 若运动了一定范围，则把点云放入地图中
    SE3 current_pose = ieskf_.GetNominalSE3();
    SE3 delta_pose   = last_pose_.inverse() * current_pose;

    if (delta_pose.translation().norm() > 1.0 || delta_pose.so3().log().norm() > math::deg2rad(10.0))
    {
        // 将地图合入NDT中
        CloudPtr current_scan_world(new PointCloudType);
        pcl::transformPointCloud(*current_scan_filter, *current_scan_world, current_pose.matrix());
        ndt_.AddCloud(current_scan_world);
        last_pose_ = current_pose;
    }

    MapIncremental(current_scan_filter, current_pose, Nearest_Points);

    // 放入UI
    if (ui_)
    {
        ui_->UpdateScan(current_scan_, current_nav_state.GetSE3());  // 转成Lidar Pose传给UI
        ui_->UpdateNavState(current_nav_state);
    }

    frame_num_++;
    return;
}

void LioIEKF::PCLCallBack(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    sync_->ProcessCloud(msg);
}

void LioIEKF::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr& msg)
{
    sync_->ProcessCloud(msg);
}

void LioIEKF::IMUCallBack(IMUPtr msg_in)
{
    sync_->ProcessIMU(msg_in);
}

void LioIEKF::Finish()
{
    if (ui_)
    {
        ui_->Quit();
    }
    LOG(INFO) << "finish done";
}

}  // namespace sad
