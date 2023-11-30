#include <json/json.h>
#include <fstream>
#include <iostream>
#include <sophus/se3.hpp>
#include <string>
#include <vector>

using namespace Sophus;

int main()
{
    Eigen::Quaterniond R_no(-0.000231837, -5.99074e-05, 0.711233, 0.70295);
    R_no.normalize();

    Eigen::Vector3d t_no(-71.2716, 119.604, -0.877343);

    SE3d pose;
    SO3d R = SO3d::rotZ(M_PI / 2.0);
    pose.setRotationMatrix(R.matrix());

    Eigen::Matrix3d RR;
    RR << 0.999139487743, -0.041377406567, 0.003021436278, 0.041374932975, 0.999143421650, 0.000883641071,
        -0.003055412322, -0.000757867470, 0.99999505281;
    Eigen::Quaterniond q(RR);
    q.normalize();
    Eigen::Vector3d tt;
    tt << 2.734453916550, -3.578369617462, -0.912014007568;
    SE3 pose2(q, tt);
    pose2 *= pose;

    Sophus::SE3d T_no(pose2);

    const std::string slot_json = "/home/holo/Downloads/hyper_matching/json.json";

    // 读取JSON文件
    std::ifstream file(slot_json);
    if (!file.is_open())
    {
        std::cerr << "Failed to open JSON file." << std::endl;
        return 1;
    }

    // 从文件读取JSON数据
    Json::Value root;
    file >> root;

    // 关闭文件
    file.close();

    std::vector<Eigen::Vector3d> all_points;

    // 从JSON中提取数据
    std::string annotation_version = root["annotation_version"].asString();
    std::string map_filename       = root["map_filename"].asString();
    std::cout << "annotation_version : " << annotation_version << std::endl;
    std::cout << "map_filename : " << map_filename << std::endl;

    // 处理parkingslots数组
    Json::Value& parkingslots = root["parkingslots"];
    for (auto& parking : parkingslots)
    {
        // 提取parking的数据
        Json::Value& vertex            = parking["vertex"];
        Json::Value  vertex_is_visible = parking["vertex_is_visible"];
        Json::Value  rod               = parking["rod"];
        Json::Value  rod_is_visible    = parking["rod_is_visible"];
        std::string  type              = parking["type"].asString();
        std::string  occupy_status     = parking["occupy_status"].asString();

        // 在这里可以根据需要处理提取到的数据
        // 例如，输出到控制台
        std::cout << "Type: " << type << std::endl;
        std::cout << "Occupy Status: " << occupy_status << std::endl;

        Eigen::Vector3d point_o;

        // 处理vertex数组
        for (int i = 0; i < vertex.size(); ++i)
        {
            double x                = vertex[i][0].asDouble();
            double y                = vertex[i][1].asDouble();
            bool   is_visible       = vertex_is_visible[i].asBool();
            point_o[0]              = x;
            point_o[1]              = y;
            point_o[2]              = 0;
            Eigen::Vector3d point_n = T_no * point_o;
            vertex[i][0]            = point_n[0];
            vertex[i][1]            = point_n[1];
            // vertex[i][2]            = point_n[2];
            all_points.emplace_back(point_n);
            std::cout << "Vertex " << i + 1 << ": (" << vertex[i][0] << ", " << vertex[i][1]
                      << "), Visible: " << is_visible << std::endl;
        }
        // parking["vertex"] = vertex;
    }

    // 写入JSON数据到文件
    std::ofstream outFile("/home/holo/Downloads/hyper_matching/new_slot.json");
    if (!outFile.is_open())
    {
        std::cerr << "Failed to open output file." << std::endl;
        return 1;
    }

    // 将JSON数据写入文件
    outFile << root;

    // 关闭文件
    outFile.close();

    std::ofstream of("/home/holo/Downloads/hyper_matching/new_slot.txt");

    for (const auto& point : all_points)
    {
        of << point[0] << " " << point[1] << " " << point[2] << std::endl;
    }
    of.close();
}
